"""JAX PPO trainer (Track B3, SCALING_PLAN.md) — PureJaxRL-structure port of
``train_ppo.train`` + SB3 PPO.

Design decisions and deliberate deviations are recorded in DECISIONS.md
(D34–D43).  Summary of the SB3 mappings implemented here:

* Network (D35): separate pi/vf MLPs ``[128, 64]`` Tanh, orthogonal init
  (gain √2 hidden / 0.01 pi-head / 1.0 vf-head), MultiDiscrete([3,3,4]) logits
  split ``[3,3,4]``; log-prob and entropy sum over the three action dims
  (verified against SB3 2.9.2 ``distributions.py:351-358``).
* Loss: ``pg = -mean(min(adv·ratio, adv·clip(ratio, 1±0.1)))``,
  ``vf = MSE(returns, values)`` (no value clipping — SB3 default),
  ``ent_loss = -mean(entropy)``, ``loss = pg + ent_coef·ent_loss + vf_coef·vf``;
  per-minibatch advantage normalisation (mean 0, std ddof=1, +1e-8).
* Optimizer (D38): ``optax.chain(clip_by_global_norm(0.5), add_decayed_weights(w),
  adam(lr, eps=1e-8))`` — coupled L2 exactly like ``torch.Adam(weight_decay=w)``;
  LR decays linearly to 0 across updates (port of ``_linear_schedule``).
* Normalisation (D26): SB3 ``RunningMeanStd`` moment algorithm (f64); obs and
  discounted-return streams updated per step over the env batch in exactly
  ``VecNormalize.step_wait`` order (update raw → normalise → returns update →
  reward norm); ``clip_obs=10``, ``clip_rew=10``, reward norm without mean
  subtraction; frozen stats + raw rewards for evaluation.
* Truncation bootstrap (SB3 ``on_policy_algorithm.py:236-245``): at
  ``truncated`` steps the buffered reward gets ``+ γ·V(normalised terminal obs)``
  (our env emits zeros as the done-obs — preserved from ``env_bracket``), and
  GAE treats every done as an episode boundary.
* target_kl (D37): ``lax.while_loop`` over minibatches; stops all further
  updates once ``approx_kl > 1.5 · target_kl``, where
  ``approx_kl = mean((e^logratio − 1) − logratio)`` from the pre-update
  log-probs — SB3's formula, threshold and stop semantics.
* Checkpoint selection (D40): separate jitted deterministic evals driven from
  Python every ``eval_freq`` steps, ``score = min(q_train, q_val)`` with the
  ``_ConsistencyEvalCallback`` eligibility rules; norm stats snapshot with every
  saved checkpoint (ARTIFACTS.md invariant 1).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import Array

import jax_env
from jax_env import EnvParams, EnvState

# Action-space geometry: MultiDiscrete([3, 3, 4]) as in env_bracket/config.
LOGIT_SLICES = (slice(0, 3), slice(3, 6), slice(6, 10))


# ── Normalisation state (D26): SB3 RunningMeanStd / VecNormalize port ───────


class NormState(NamedTuple):
    obs_mean: Array    # f64 [obs_dim]
    obs_var: Array     # f64 [obs_dim]
    obs_count: Array   # f64 scalar
    ret_mean: Array    # f64 scalar
    ret_var: Array     # f64 scalar
    ret_count: Array   # f64 scalar


def init_norm_state(obs_dim: int) -> NormState:
    """SB3 RunningMeanStd init: mean 0, var 1, count epsilon=1e-4."""
    return NormState(
        obs_mean=jnp.zeros((obs_dim,), dtype=jnp.float64),
        obs_var=jnp.ones((obs_dim,), dtype=jnp.float64),
        obs_count=jnp.asarray(1e-4, dtype=jnp.float64),
        ret_mean=jnp.asarray(0.0, dtype=jnp.float64),
        ret_var=jnp.asarray(1.0, dtype=jnp.float64),
        ret_count=jnp.asarray(1e-4, dtype=jnp.float64),
    )


def _update_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """SB3 ``RunningMeanStd.update_from_moments`` (exact op order)."""
    delta = batch_mean - mean
    tot = count + batch_count
    new_mean = mean + delta * batch_count / tot
    m_a = var * count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + jnp.square(delta) * count * batch_count / (count + batch_count)
    new_var = m_2 / tot
    return new_mean, new_var, count + batch_count


def update_obs_rms(norm: NormState, obs: Array) -> NormState:
    """``VecNormalize.step_wait``/``reset``: update with the RAW obs."""
    batch = obs.astype(jnp.float64)
    m, v, c = _update_from_moments(
        norm.obs_mean, norm.obs_var, norm.obs_count,
        batch.mean(axis=0), batch.var(axis=0), batch.shape[0],
    )
    return norm._replace(obs_mean=m, obs_var=v, obs_count=c)


def update_ret_rms(norm: NormState, returns: Array) -> NormState:
    batch = returns.astype(jnp.float64)
    m, v, c = _update_from_moments(
        norm.ret_mean, norm.ret_var, norm.ret_count,
        batch.mean(), batch.var(), batch.shape[0],
    )
    return norm._replace(ret_mean=m, ret_var=v, ret_count=c)


def normalize_obs(obs: Array, norm: NormState) -> Array:
    """``clip((obs - mean) / sqrt(var + 1e-8), ±10)`` (SB3 vec_normalize:221)."""
    out = (obs.astype(jnp.float64) - norm.obs_mean) / jnp.sqrt(norm.obs_var + 1e-8)
    return jnp.clip(out, -10.0, 10.0).astype(jnp.float32)


def normalize_reward(reward: Array, norm: NormState) -> Array:
    """``clip(reward / sqrt(ret_var + 1e-8), ±10)`` — no mean subtraction."""
    out = reward.astype(jnp.float64) / jnp.sqrt(norm.ret_var + 1e-8)
    return jnp.clip(out, -10.0, 10.0).astype(jnp.float32)


# ── Network (D35) — batched: obs [B, obs_dim] ───────────────────────────────


def _orthogonal(key, shape, gain):
    """Orthogonal init on the HOST (numpy QR) — DECISIONS.md D45.

    JAX's ``initializers.orthogonal`` runs its QR through cuSolver, which is
    unreliable on this Blackwell part (sm_120): ``cusolverDnDgeqrf`` raised
    INTERNAL errors mid-suite and poisoned the context.  numpy's QR gives the
    same distribution with rows/columns orthonormal on the smaller side.
    """
    n_rows, n_cols = shape
    a = np.asarray(jax.random.normal(key, (max(n_rows, n_cols), min(n_rows, n_cols))))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    w = q.T if n_rows < n_cols else q
    return jnp.asarray(w[:n_rows, :n_cols] * gain, dtype=jnp.float32)


def init_params(rng: Array, obs_dim: int) -> dict:
    """Orthogonal init matching SB3's ``module_gains`` (D35)."""

    def dense(rng, in_dim, out_dim, gain):
        w = _orthogonal(rng, (in_dim, out_dim), gain)
        return {"w": w.astype(jnp.float32), "b": jnp.zeros((out_dim,), dtype=jnp.float32)}

    k = jax.random.split(rng, 6)
    return {
        "pi": {
            "dense0": dense(k[0], obs_dim, 128, np.sqrt(2.0)),
            "dense1": dense(k[1], 128, 64, np.sqrt(2.0)),
            "head": dense(k[2], 64, 10, 0.01),
        },
        "vf": {
            "dense0": dense(k[3], obs_dim, 128, np.sqrt(2.0)),
            "dense1": dense(k[4], 128, 64, np.sqrt(2.0)),
            "head": dense(k[5], 64, 1, 1.0),
        },
    }


def _mlp_apply(block: dict, obs: Array) -> Array:
    x = jnp.tanh(obs @ block["dense0"]["w"] + block["dense0"]["b"])
    x = jnp.tanh(x @ block["dense1"]["w"] + block["dense1"]["b"])
    return x @ block["head"]["w"] + block["head"]["b"]


def policy_logits(params: dict, obs: Array) -> Array:
    """obs [B, obs_dim] -> logits [B, 10]."""
    return _mlp_apply(params["pi"], obs.astype(jnp.float32))


def value_fn(params: dict, obs: Array) -> Array:
    """obs [B, obs_dim] -> values [B]."""
    return _mlp_apply(params["vf"], obs.astype(jnp.float32)).squeeze(-1)


def action_log_prob(params: dict, obs: Array, actions: Array) -> Array:
    """MultiDiscrete log-prob [B]: sum of per-dim categorical log-probs (D43)."""
    logits = policy_logits(params, obs)
    idx = jnp.arange(logits.shape[0])
    total = jnp.zeros((logits.shape[0],), dtype=jnp.float32)
    for k, sl in enumerate(LOGIT_SLICES):
        logp = jax.nn.log_softmax(logits[:, sl], axis=-1)
        total = total + logp[idx, actions[:, k].astype(jnp.int32)]
    return total


def action_entropy(params: dict, obs: Array) -> Array:
    """[B] sum of per-dim categorical entropies (SB3 MultiCategorical.entropy)."""
    logits = policy_logits(params, obs)
    total = jnp.zeros((logits.shape[0],), dtype=jnp.float32)
    for sl in LOGIT_SLICES:
        logp = jax.nn.log_softmax(logits[:, sl], axis=-1)
        total = total - jnp.sum(jnp.exp(logp) * logp, axis=-1)
    return total


def sample_actions(rng: Array, logits: Array) -> Array:
    """Per-dim categorical sample for ONE env (logits [10], 3 rng keys)."""
    keys = jax.random.split(rng, 3)
    acts = [jax.random.categorical(keys[k], logits[sl])
            for k, sl in enumerate(LOGIT_SLICES)]
    return jnp.stack(acts).astype(jnp.int32)


def deterministic_actions(logits: Array) -> Array:
    """Batched per-dim argmax (SB3 ``MultiCategorical.mode``): [B, 10] -> [B, 3]."""
    return jnp.stack([jnp.argmax(logits[:, sl], axis=-1)
                      for sl in LOGIT_SLICES], axis=1).astype(jnp.int32)


# ── GAE (D39: unit-tested against a numpy reference) ────────────────────────


def gae(rewards: Array, values: Array, dones: Array, last_values: Array,
        gamma: float = 0.99, gae_lambda: float = 0.95):
    """SB3 ``RolloutBuffer.compute_returns_and_advantage`` semantics, f32.

    ``dones[t]`` marks that the obs at t+1 starts a fresh episode (SB3's
    ``episode_starts[step + 1]``); the truncation bootstrap has already been
    folded into ``rewards`` by the rollout.
    """
    rewards = rewards.astype(jnp.float32)
    values = values.astype(jnp.float32)
    non_term = 1.0 - dones.astype(jnp.float32)
    last_values = last_values.astype(jnp.float32)

    def body(carry, xs):
        last_gae, next_value = carry
        r, v, nt = xs
        delta = r + gamma * next_value * nt - v
        gae_t = delta + gamma * gae_lambda * nt * last_gae
        return (gae_t, v), gae_t

    _, advantages = jax.lax.scan(
        body,
        (jnp.zeros_like(last_values), last_values),
        (rewards, values, non_term),
        reverse=True,
    )
    return advantages, advantages + values


# ── PPO update (D37/D38) ────────────────────────────────────────────────────


def _make_optimizer(lr, weight_decay: float):
    import optax
    return optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.add_decayed_weights(weight_decay),
        optax.adam(learning_rate=lr, eps=1e-8),
    )


def make_optimizer(lr: float, weight_decay: float):
    """Public builder — the driver uses it to initialise ``opt_state``."""
    return _make_optimizer(lr, weight_decay)


def _minibatch_loss(params, obs, actions, old_logp, returns, adv, clip_range,
                    ent_coef, vf_coef):
    logp = action_log_prob(params, obs, actions)
    ratio = jnp.exp(logp - old_logp)
    adv_n = (adv - adv.mean()) / (adv.std(ddof=1) + 1e-8)   # SB3 torch std(): ddof=1
    pg_loss = -jnp.minimum(adv_n * ratio,
                           adv_n * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)).mean()
    v_pred = value_fn(params, obs)
    vf_loss = jnp.mean(jnp.square(returns - v_pred))
    ent = action_entropy(params, obs)
    ent_loss = -ent.mean()
    loss = pg_loss + ent_coef * ent_loss + vf_coef * vf_loss
    return loss, (pg_loss, vf_loss, ent_loss, logp)


def make_update_fn(num_minibatches: int, n_epochs: int, clip_range: float,
                   ent_coef: float, vf_coef: float, target_kl: float | None,
                   weight_decay: float):
    """Jitted ``update(params, opt_state, traj, rng, lr)``.

    ``traj``: flattened (T·N) arrays obs/actions/old_logp/returns/advantages.
    Epoch loop → per-epoch permutation → ``lax.while_loop`` over minibatches
    with the D37 stop semantics; returns (params, opt_state, metrics).
    """

    def update(params, opt_state, traj, rng, lr):
        opt = _make_optimizer(lr, weight_decay)
        batch_size = traj["obs"].shape[0]
        mb_size = batch_size // num_minibatches

        perm_rngs = jax.random.split(rng, n_epochs)
        perms = jax.vmap(lambda k: jax.random.permutation(k, batch_size))(perm_rngs)
        shuffled = jax.tree_util.tree_map(
            lambda x: jax.vmap(lambda p: jnp.take(x, p, axis=0))(perms)
                      .reshape(n_epochs, num_minibatches, mb_size, *x.shape[1:]),
            traj)

        def epoch_body(carry, epoch_batch):
            params, opt_state = carry

            def mb_body(mb_carry):
                params, opt_state, idx, stopped, pg_s, vf_s, ent_s, kl_s, cnt = mb_carry
                mb = jax.tree_util.tree_map(lambda x: x[idx], epoch_batch)

                def do_update(_):
                    (_loss, (pg, vf, ent, logp)), grads = jax.value_and_grad(
                        _minibatch_loss, has_aux=True)(
                        params, mb["obs"], mb["actions"], mb["old_logp"],
                        mb["returns"], mb["advantages"],
                        clip_range, ent_coef, vf_coef)
                    updates, opt_state2 = opt.update(grads, opt_state, params)
                    params2 = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
                    log_ratio = logp - mb["old_logp"]
                    approx_kl = jnp.mean((jnp.exp(log_ratio) - 1.0) - log_ratio)
                    if target_kl is not None:
                        stop = approx_kl > 1.5 * target_kl
                    else:
                        stop = jnp.asarray(False)
                    return (params2, opt_state2, idx + 1, stopped | stop,
                            pg_s + pg, vf_s + vf, ent_s + ent, kl_s + approx_kl, cnt + 1)

                def skip(_):
                    return (params, opt_state, idx + 1, stopped,
                            pg_s, vf_s, ent_s, kl_s, cnt)

                return jax.lax.cond(stopped, skip, do_update, operand=None)

            init = (params, opt_state, jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(False, dtype=bool),
                    jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32),
                    jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32),
                    jnp.zeros((), dtype=jnp.float32))
            params, opt_state, _idx, _stopped, pg_s, vf_s, ent_s, kl_s, cnt = \
                jax.lax.while_loop(lambda c: (c[2] < num_minibatches) & (~c[3]),
                                   mb_body, init)
            return (params, opt_state), (pg_s, vf_s, ent_s, kl_s, cnt)

        (params, opt_state), metrics = jax.lax.scan(
            epoch_body, (params, opt_state), shuffled)
        pg, vf, ent, kl, cnt = metrics
        tot = jnp.maximum(cnt.sum(), 1.0)
        agg = {
            "pg_loss": pg.sum() / tot,
            "vf_loss": vf.sum() / tot,
            "entropy_loss": ent.sum() / tot,
            "approx_kl": kl.sum() / tot,
            "n_updates": cnt.sum(),
        }
        return params, opt_state, agg

    return jax.jit(update)


# ── Rollout (SB3 VecNormalize + collect_rollouts semantics) ─────────────────


class TrainState(NamedTuple):
    params: dict
    opt_state: Any
    norm: NormState
    returns: Array        # f64 [num_envs] — VecNormalize discounted-return state
    env_state: EnvState   # batched [num_envs]
    obs_norm: Array       # f32 [num_envs, obs_dim] — current normalised obs
    rng: Array
    update_idx: Array     # i32


def make_rollout_fn(num_envs: int, num_steps: int, gamma: float):
    """Jitted ``rollout(train_state, env_params) -> (train_state, traj)``.

    Per step (SB3 ``step_wait`` order): act on the carried normalised obs →
    env step (auto-reset re-randomises starts like the SB3 training envs) →
    obs_rms update with raw obs → normalise → returns update → ret_rms update →
    reward norm → truncation bootstrap ``+γ·V(norm terminal obs)`` → zero the
    returns accumulator at dones.  ``env_params`` is a function argument (JIT
    hygiene, D34) — fold data never enters the closure.
    """

    def rollout(train_state: TrainState, env_params: EnvParams):
        def body(carry, _):
            ts = carry
            rng, sample_rng, reset_rng = jax.random.split(ts.rng, 3)
            logits = policy_logits(ts.params, ts.obs_norm)
            sample_rngs = jax.random.split(sample_rng, num_envs)
            actions = jax.vmap(sample_actions)(sample_rngs, logits)
            logp = action_log_prob(ts.params, ts.obs_norm, actions)
            values = value_fn(ts.params, ts.obs_norm)

            reset_rngs = jax.random.split(reset_rng, num_envs)
            obs_next, env_state, rew_raw, term, trunc, info, final_obs, _ = jax.vmap(
                jax_env.step_autoreset, in_axes=(0, 0, 0, None))(
                reset_rngs, ts.env_state, actions, env_params)
            done = term | trunc

            norm = update_obs_rms(ts.norm, obs_next)
            obs_next_norm = normalize_obs(obs_next, norm)
            returns = ts.returns * gamma + rew_raw
            norm = update_ret_rms(norm, returns)
            rew_norm = normalize_reward(rew_raw, norm)

            v_term = value_fn(ts.params, normalize_obs(final_obs, norm))
            rew_buffer = rew_norm + jnp.where(trunc, gamma * v_term, 0.0)
            returns = jnp.where(done, 0.0, returns)

            new_ts = ts._replace(norm=norm, returns=returns, env_state=env_state,
                                 obs_norm=obs_next_norm, rng=rng)
            trans = {
                "obs": ts.obs_norm.astype(jnp.float32),
                "actions": actions.astype(jnp.int32),
                "old_logp": logp.astype(jnp.float32),
                "values": values.astype(jnp.float32),
                "rewards": rew_buffer.astype(jnp.float32),
                "dones": done,
            }
            return new_ts, trans

        ts, traj = jax.lax.scan(body, train_state, None, length=num_steps)
        traj["last_values"] = value_fn(ts.params, ts.obs_norm).astype(jnp.float32)
        return ts, traj

    return jax.jit(rollout)


def make_trajectory_postprocess(num_steps: int, num_envs: int, gamma: float,
                                gae_lambda: float):
    """Jitted GAE over a scanned trajectory; flattens (T·N) for the update."""

    def post(traj):
        adv, ret = gae(traj["rewards"], traj["values"], traj["dones"],
                       traj["last_values"], gamma, gae_lambda)
        flat = lambda x: x.reshape(num_steps * num_envs, *x.shape[2:])
        return {
            "obs": flat(traj["obs"]),
            "actions": flat(traj["actions"]),
            "old_logp": flat(traj["old_logp"]),
            "returns": flat(ret.astype(jnp.float32)),
            "advantages": flat(adv.astype(jnp.float32)),
        }

    return jax.jit(post)


# ── Deterministic eval episode (D40) ────────────────────────────────────────


def _eval_fn_impl(params, norm: NormState, env_params: EnvParams):
    """Deterministic episode from index 0; frozen norm stats; raw rewards.

    Returns per-step reward, equity and account state plus up-to-two trade
    records per step — enough for ``_ConsistencyEvalCallback``'s
    (reward, max_dd_pct, n_trades) and for ``rollout_on_split``'s frames.
    ``env_params`` is an argument (D34).
    """
    length = env_params.close.shape[0] - 2
    state = jax_env.init_state(env_params, 0)
    obs_raw = jax_env.observation(state, env_params)[None, :]
    obs_norm = normalize_obs(obs_raw, norm)

    def body(carry, _):
        state, obs_norm = carry
        action = deterministic_actions(policy_logits(params, obs_norm))[0]
        hist_close = env_params.close[state.i]
        hist_ts = env_params.decision_ts[state.i]
        obs_next, state, rew, term, trunc, info = jax_env.step(state, action, env_params)
        done = term | trunc
        obs_next = jnp.where(done, jnp.zeros_like(obs_next), obs_next)
        obs_next_norm = normalize_obs(obs_next[None, :], norm)
        return (state, obs_next_norm), {
            "reward": rew,
            "equity": info.equity,
            "realized": info.realized,
            "direction": info.direction,
            "close": hist_close,
            "time_ns": hist_ts,
            "closed": info.closed,
            "reason": info.reason,
            "trade_direction": info.trade_direction,
            "entry_ts": info.entry_ts,
            "exit_ts": info.exit_ts,
            "entry_price": info.entry_price,
            "exit_price": info.exit_price,
            "sl": info.sl,
            "tp": info.tp,
            "sl_atr_mult": info.sl_atr_mult,
            "tp_r": info.tp_r,
            "units": info.units,
            "pnl": info.pnl,
            "r_mult": info.r_mult,
            "bars_in_trade": info.bars_in_trade,
        }

    _carry, out = jax.lax.scan(body, (state, obs_norm), None, length=length)
    return out


eval_fn = jax.jit(_eval_fn_impl)


def make_eval_fn():
    """Singleton jitted eval (shape-keyed cache; never rebuild per call)."""
    return eval_fn


# ── Checkpoint IO (D42) ─────────────────────────────────────────────────────


def save_checkpoint(path: Path | str, params: dict, norm: NormState) -> None:
    """Flat npz: network leaves as ``pi/dense0/w``-style keys + norm arrays.

    Params and norm stats always travel in ONE file (ARTIFACTS.md invariant 1).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}

    def walk(prefix, node):
        for k, v in node.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                walk(key, v)
            else:
                arrays[key] = np.asarray(v)

    walk("", params)
    for field in NormState._fields:
        arrays[f"norm/{field}"] = np.asarray(getattr(norm, field))
    np.savez_compressed(path, **arrays)


def save_norm_snapshot(path: Path | str, norm: NormState) -> None:
    """Norm-only companion file (keeps the ARTIFACTS run_info key semantics)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f: np.asarray(getattr(norm, f))
                                 for f in NormState._fields})


def load_checkpoint(path: Path | str) -> tuple[dict, NormState]:
    data = np.load(Path(path))
    params: dict = {"pi": {}, "vf": {}}
    norm_arrays = {}
    for key in data.files:
        if key.startswith("norm/"):
            norm_arrays[key.split("/", 1)[1]] = jnp.asarray(data[key])
        else:
            net, layer, name = key.split("/")
            params[net].setdefault(layer, {})[name] = jnp.asarray(data[key])
    return params, NormState(**norm_arrays)


# ── B4: uniform model handle + split rollout (driver/loader integration) ────


class JaxModel(NamedTuple):
    """Loader handle: a checkpointed policy + its norm stats.

    Consumed by the same call sites as an SB3 ``PPO`` object via the framework
    branches in ``train_ppo._load_fold_model`` / ``train_ppo._rollout_on_split``
    and the loader branches in ``final_holdout_eval.py`` /
    ``training_diagnostics.py`` (SCALING_PLAN.md B4).
    """
    params: dict
    norm: NormState


def load_model(path: Path | str) -> JaxModel:
    params, norm = load_checkpoint(path)
    return JaxModel(params=params, norm=norm)


def slice_m1_for_decision_window(m1: pd.DataFrame, decision_df: pd.DataFrame) -> pd.DataFrame:
    """Local copy of ``train_ppo._slice_m1_for_decision_window`` (avoids a
    circular import; the rule must stay in sync with that function)."""
    if decision_df.empty:
        return m1.iloc[0:0].copy()
    start, end = decision_df.index.min(), decision_df.index.max()
    return m1.loc[(m1.index > start) & (m1.index <= end)].copy()


def build_env_params_cfg(decision_df, m1_sl, feature_cols, episode_steps=None) -> EnvParams:
    """``build_env_params`` with ``config.CFG`` trade economics — bit-compatible
    with ``train_ppo.build_env``'s kwargs."""
    from config import CFG
    return jax_env.build_env_params(
        decision_df, m1_sl, feature_cols,
        sl_atr_multipliers=CFG.sl_atr_multipliers,
        tp_r_multipliers=CFG.tp_r_multipliers,
        initial_equity=CFG.initial_equity,
        risk_fraction=CFG.risk_fraction,
        spread_price=CFG.spread_price,
        slippage_price=CFG.slippage_price,
        commission_per_trade=CFG.commission_per_trade,
        holding_penalty=CFG.holding_penalty,
        reward_mtm_weight=CFG.reward_mtm_weight,
        max_episode_steps=episode_steps,
    )


def rollout_on_split(handle: JaxModel, m1: pd.DataFrame, feature_cols: list[str],
                     decision_df: pd.DataFrame):
    """Deterministic rollout over one decision split — JAX twin of
    ``train_ppo._rollout_on_split``.

    Returns ``(equity_df, trades_df, report_dict)`` with the exact schemas of
    ``BracketTradingEnv.equity_curve()`` / ``trade_log()`` so every downstream
    consumer (evaluate.py, stitching, CSVs) is reused unchanged.
    """
    from evaluate import full_report
    from config import CFG

    m1_sl = slice_m1_for_decision_window(m1, decision_df)
    env_params = build_env_params_cfg(decision_df, m1_sl, feature_cols, episode_steps=None)
    out = make_eval_fn()(handle.params, handle.norm, env_params)

    idx = pd.Index(
        jax_env.stamps_from_ts(np.asarray(out["time_ns"]), decision_df.index), name="time")
    eq = pd.DataFrame({
        "equity": np.asarray(out["equity"], dtype=np.float64),
        "realized_pnl": np.asarray(out["realized"], dtype=np.float64),
        "position": np.asarray(out["direction"], dtype=np.int8),
        "close": np.asarray(out["close"], dtype=np.float64),
        "reward": np.asarray(out["reward"], dtype=np.float64),
    }, index=idx)

    rows = []
    closed = np.asarray(out["closed"])          # [steps, 2] — time-major scan out
    for t in range(closed.shape[0]):
        for slot in (0, 1):
            if bool(closed[t, slot]):
                rows.append({
                    "entry_time": jax_env.stamps_from_ts([int(out["entry_ts"][t, slot])], decision_df.index)[0],
                    "exit_time": jax_env.stamps_from_ts([int(out["exit_ts"][t, slot])], decision_df.index)[0],
                    "direction": int(out["trade_direction"][t, slot]),
                    "entry_price": float(out["entry_price"][t, slot]),
                    "exit_price": float(out["exit_price"][t, slot]),
                    "sl": float(out["sl"][t, slot]),
                    "tp": float(out["tp"][t, slot]),
                    "sl_atr_mult": float(out["sl_atr_mult"][t, slot]),
                    "tp_r_bracket": float(out["tp_r"][t, slot]),
                    "units": float(out["units"][t, slot]),
                    "pnl": float(out["pnl"][t, slot]),
                    "r_mult": float(out["r_mult"][t, slot]),
                    "bars_in_trade": int(out["bars_in_trade"][t, slot]),
                    "exit_reason": jax_env.REASON_STRINGS[int(out["reason"][t, slot])],
                })
    trades = pd.DataFrame(rows, columns=[
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "sl", "tp", "sl_atr_mult", "tp_r_bracket", "units", "pnl", "r_mult",
        "bars_in_trade", "exit_reason",
    ])
    rep = full_report(eq, trades, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()
    return eq, trades, rep


def post_training_eval_jax(handle: JaxModel, m1: pd.DataFrame, feature_cols: list[str],
                           splits: dict, out_dir: Path | str, slug: str) -> Path:
    """JAX twin of ``train_ppo._post_training_eval`` (A6): chained in-sample →
    out-of-sample equity chart from deterministic rollouts."""
    from visualize import plot_insample_oos_equity, save_html
    from config import CFG

    equities: dict = {}
    for name, decision_df in splits.items():
        if decision_df is None or decision_df.empty:
            continue
        eq, trades, _rep = rollout_on_split(handle, m1, feature_cols, decision_df)
        equities[name] = eq
        final = float(eq["equity"].iloc[-1]) if len(eq) else CFG.initial_equity
        ret = (final / CFG.initial_equity - 1) * 100
        print(f"  {name:<6}  trades={len(trades):>4}  "
              f"final=${final:>9,.0f}  return={ret:+.1f}%")

    fig = plot_insample_oos_equity(
        equities, initial_equity=CFG.initial_equity,
        title=f"Equity — in-sample vs out-of-sample  [{slug}]",
    )
    out_path = Path(out_dir) / f"{slug}_equity_insample_oos.html"
    save_html(fig, out_path)
    print(f"\nEquity chart → {out_path}")
    return out_path


# ── Trainer driver ──────────────────────────────────────────────────────────


def train_jax(
    total_timesteps: int = 3_000_000,
    seed: int = 42,
    out_dir: str = "models",
    train_episode_steps: int = 2048,
    eval_freq: int = 50_000,
    dd_penalty: float = 1.0,
    datasets: tuple | None = None,
    num_envs: int | None = None,
    num_steps: int | None = None,
    num_minibatches: int | None = None,
    post_eval_chart: bool = False,
    reveal_test: bool = False,
    learning_rate: float | None = None,
    ent_coef: float | None = None,
    clip_range: float | None = None,
    log_fn=print,
):
    """One training run over a train/val split — the B3 entry point.

    Hyperparameters come from ``config.CFG`` (D36 mapping); geometry defaults
    to ``CFG.jax_num_envs × CFG.jax_num_steps`` with the SB3-parity preset
    available by overriding.  B4 calls this per fold (``datasets`` tuple);
    the hp overrides exist for the B5 swarm sweeps.
    """
    from config import CFG
    import train_ppo as tp

    num_envs = CFG.jax_num_envs if num_envs is None else num_envs
    num_steps = CFG.jax_num_steps if num_steps is None else num_steps
    num_minibatches = (CFG.jax_num_minibatches if num_minibatches is None
                       else num_minibatches)
    gamma, gae_lambda = 0.99, 0.95
    clip_range = CFG.ppo_clip_range if clip_range is None else clip_range
    ent_coef = CFG.ppo_ent_coef if ent_coef is None else ent_coef
    vf_coef = 0.5
    target_kl = CFG.ppo_target_kl
    weight_decay = CFG.ppo_weight_decay
    n_epochs = CFG.ppo_n_epochs
    lr0 = CFG.ppo_learning_rate if learning_rate is None else learning_rate

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        m1, feature_cols, train_feat, val_feat, test_feat = tp.load_datasets()
    else:
        m1, feature_cols, train_feat, val_feat, test_feat = datasets
    train_m1 = tp._slice_m1_for_decision_window(m1, train_feat)
    val_m1 = tp._slice_m1_for_decision_window(m1, val_feat)
    train_eval_feat = train_feat.iloc[-len(val_feat):]
    train_eval_m1 = tp._slice_m1_for_decision_window(m1, train_eval_feat)

    obs_dim = len(feature_cols) + 6

    def build(df, m1_sl, episode_steps):
        return build_env_params_cfg(df, m1_sl, feature_cols, episode_steps)

    env_train = build(train_feat, train_m1, train_episode_steps)
    env_train_eval = build(train_eval_feat, train_eval_m1, None)
    env_val = build(val_feat, val_m1, None)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, reset_rng = jax.random.split(rng, 3)
    params = init_params(init_rng, obs_dim)
    norm = init_norm_state(obs_dim)

    # Initial env reset (VecNormalize.reset semantics: update rms with the raw
    # obs, then normalise).
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obs_raw, env_state = jax.vmap(jax_env.reset, in_axes=(0, None))(reset_rngs, env_train)
    norm = update_obs_rms(norm, obs_raw)
    obs_norm = normalize_obs(obs_raw, norm)

    opt = make_optimizer(lr0, weight_decay)
    opt_state = opt.init(params)

    state = TrainState(params=params, opt_state=opt_state, norm=norm,
                       returns=jnp.zeros((num_envs,), dtype=jnp.float64),
                       env_state=env_state, obs_norm=obs_norm, rng=rng,
                       update_idx=jnp.asarray(0, dtype=jnp.int32))

    rollout_fn = make_rollout_fn(num_envs, num_steps, gamma)
    post_fn = make_trajectory_postprocess(num_steps, num_envs, gamma, gae_lambda)
    update_fn = make_update_fn(num_minibatches, n_epochs, clip_range, ent_coef,
                               vf_coef, target_kl, weight_decay)

    buffer = num_envs * num_steps
    num_updates = max(int(np.ceil(total_timesteps / buffer)), 1)
    eval_every = max(int(np.ceil(eval_freq / buffer)), 1)
    log_fn(f"[jax] geometry: {num_envs} envs x {num_steps} steps = {buffer} "
           f"transitions/update, {num_updates} updates, eval every {eval_every} updates")

    # Consistency-eval bookkeeping (D40 / _ConsistencyEvalCallback port).
    best_score = -float("inf")
    eval_rows: list[dict] = []
    t0 = time.time()

    def run_leg(env_params):
        out = eval_fn(state.params, state.norm, env_params)
        reward = float(np.asarray(out["reward"]).sum())
        eq = np.asarray(out["equity"], dtype=np.float64)
        peak = np.maximum.accumulate(eq)
        max_dd_pct = float(np.abs((eq / peak - 1.0).min()) * 100.0) if len(eq) else 0.0
        n_trades = int(np.asarray(out["closed"]).sum())
        return reward, max_dd_pct, n_trades

    for k in range(num_updates):
        lr = lr0 * (1.0 - k / num_updates) if CFG.ppo_lr_schedule == "linear" else lr0
        state, traj = rollout_fn(state, env_train)
        traj = post_fn(traj)
        params, opt_state, metrics = update_fn(state.params, state.opt_state, traj,
                                               state.rng, jnp.asarray(lr, dtype=jnp.float32))
        state = state._replace(params=params, opt_state=opt_state,
                               update_idx=state.update_idx + 1)
        timesteps = (k + 1) * buffer

        if (k + 1) % eval_every == 0 or k == num_updates - 1:
            train_r, train_dd, train_n = run_leg(env_train_eval)
            val_r, val_dd, val_n = run_leg(env_val)
            q_train = train_r - dd_penalty * train_dd
            q_val = val_r - dd_penalty * val_dd
            score = min(q_train, q_val)
            gap = abs(train_r - val_r)
            eligible = (train_r > 0 and val_r > 0
                        and train_n >= 5 and val_n >= 5)
            row = dict(timesteps=timesteps,
                       train_eval_r=round(train_r, 3), val_r=round(val_r, 3),
                       train_dd_pct=round(train_dd, 3), val_dd_pct=round(val_dd, 3),
                       q_train=round(q_train, 3), q_val=round(q_val, 3),
                       score=round(score, 3), gap=round(gap, 3), eligible=bool(eligible))
            eval_rows.append(row)
            marker = ""
            if eligible and score > best_score:
                best_score = score
                # Snapshot the checkpoint AND the norm stats it was selected
                # under (ARTIFACTS.md invariant 1) at this moment.
                save_checkpoint(out_path / "best_model" / "best_model.npz",
                                state.params, state.norm)
                save_norm_snapshot(out_path / "best_model" / "best_model_norm.npz",
                                   state.norm)
                marker = "  <- BEST"
            flag = "" if eligible else "  (ineligible)"
            log_fn(f"[{timesteps:>9,}]  train={train_r:+7.2f}/dd{train_dd:4.1f}%  "
                   f"val={val_r:+7.2f}/dd{val_dd:4.1f}%  score={score:+8.2f}{flag}{marker}  "
                   f"kl={float(metrics['approx_kl']):.4f}  ({time.time() - t0:.0f}s)")

            log_path = out_path / "eval_logs"
            log_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(eval_rows).to_csv(log_path / "consistency_evals.csv", index=False)

    # Final artifacts (ARTIFACTS.md layout; framework key per D42).
    slug = f"jax_ppo_{CFG.decision_timeframe}_{total_timesteps // 1000}k_seed{seed}"
    save_checkpoint(out_path / f"{slug}.npz", state.params, state.norm)
    save_norm_snapshot(out_path / f"{slug}_norm.npz", state.norm)
    run_info = {
        "framework": "jax",
        "slug": slug,
        "model_path": str(out_path / f"{slug}.npz"),
        "vecnorm_path": str(out_path / f"{slug}_norm.npz"),
        "decision_timeframe": CFG.decision_timeframe,
        "sl_atr_multipliers": list(CFG.sl_atr_multipliers),
        "tp_r_multipliers": list(CFG.tp_r_multipliers),
        "total_timesteps": total_timesteps,
        "train_episode_steps": train_episode_steps,
        "seed": seed,
        "dd_penalty": dd_penalty,
        "risk_fraction": CFG.risk_fraction,
        "spread_price": CFG.spread_price,
        "jax_num_envs": num_envs,
        "jax_num_steps": num_steps,
    }
    best = out_path / "best_model" / "best_model.npz"
    if best.exists():
        run_info["best_model_path"] = str(best)
        run_info["best_model_vecnorm_path"] = str(out_path / "best_model" / "best_model_norm.npz")
    (out_path / "run_info.json").write_text(json.dumps(run_info, indent=2))

    if post_eval_chart:
        handle = JaxModel(params=state.params, norm=state.norm)
        splits = {"Train": train_feat, "Val": val_feat}
        if reveal_test:
            splits["Test"] = test_feat
        else:
            log_fn("[jax] post-training chart: test split remains sealed")
        post_training_eval_jax(handle, m1, feature_cols, splits, out_path, slug)

    log_fn(f"[jax] done in {time.time() - t0:.0f}s -> {out_path / 'run_info.json'}")
    return state, eval_rows
