"""Track B3 unit tests — trainer-math equivalence (DECISIONS.md D39).

Three drift-prone surfaces are pinned against independent references:
  * GAE vs a numpy transcription of SB3's ``compute_returns_and_advantage``,
  * one full PPO update vs a torch recomputation of the same loss/optimizer,
  * the normalisation port vs SB3's own ``RunningMeanStd``/``VecNormalize``.

Plus a tiny end-to-end ``train_jax`` smoke that checks the artifact contract
(ARTIFACTS.md / D40 / D42).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path setup)

jax = pytest.importorskip("jax")
jnp = jax.numpy

import jax_ppo  # noqa: E402
from jax_ppo import (  # noqa: E402
    NormState, action_entropy, action_log_prob, deterministic_actions,
    gae, init_norm_state, init_params, make_optimizer, make_update_fn,
    normalize_obs, normalize_reward, policy_logits, value_fn,
)


# ── GAE vs SB3 transcription ────────────────────────────────────────────────


def _gae_reference(rewards, values, dones, last_values, gamma, lam):
    """Numpy transcription of SB3 buffers.py:403-434 (f32 semantics)."""
    rewards = rewards.astype(np.float32)
    values = values.astype(np.float32)
    non_term = 1.0 - dones.astype(np.float32)
    last_values = last_values.astype(np.float32)
    t_len, n = rewards.shape
    adv = np.zeros_like(rewards)
    last = np.zeros((n,), dtype=np.float32)
    for t in reversed(range(t_len)):
        if t == t_len - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        nt = non_term[t]
        delta = rewards[t] + gamma * next_values * nt - values[t]
        last = delta + gamma * lam * nt * last
        adv[t] = last
    return adv, adv + values


def test_gae_matches_numpy_reference():
    rng = np.random.default_rng(0)
    T, N = 37, 5
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = (rng.random((T, N)) < 0.1).astype(np.float32)
    last_values = rng.normal(size=(N,)).astype(np.float32)
    gamma, lam = 0.99, 0.95

    adv_ref, ret_ref = _gae_reference(rewards, values, dones, last_values, gamma, lam)
    adv, ret = gae(jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(dones),
                   jnp.asarray(last_values), gamma, lam)
    np.testing.assert_allclose(np.asarray(adv), adv_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ret), ret_ref, rtol=1e-5, atol=1e-6)


# ── single PPO update vs torch ──────────────────────────────────────────────


def test_single_update_matches_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    torch.manual_seed(0)
    obs_dim, B = 31, 64
    lr, wd = 6e-5, 1e-5

    params = init_params(jax.random.PRNGKey(1), obs_dim)
    rng = np.random.default_rng(2)
    obs = rng.normal(size=(B, obs_dim)).astype(np.float32)
    actions = np.column_stack([
        rng.integers(0, 3, size=B), rng.integers(0, 3, size=B),
        rng.integers(0, 4, size=B)]).astype(np.int32)
    old_logp = rng.normal(size=(B,)).astype(np.float32)
    returns = rng.normal(size=(B,)).astype(np.float32)
    adv = rng.normal(size=(B,)).astype(np.float32)
    traj = {
        "obs": jnp.asarray(obs), "actions": jnp.asarray(actions),
        "old_logp": jnp.asarray(old_logp), "returns": jnp.asarray(returns),
        "advantages": jnp.asarray(adv),
    }

    # ── JAX: standalone loss + gradients at the same point ──────────────
    (_loss, _aux), grads = jax.value_and_grad(jax_ppo._minibatch_loss, has_aux=True)(
        params, traj["obs"], traj["actions"], traj["old_logp"],
        traj["returns"], traj["advantages"], 0.1, 0.03, 0.5)
    opt = make_optimizer(lr, wd)

    # ── torch: same net, same loss, torch-Adam(wd) semantics ─────────────
    class Net(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p0 = nn.Linear(obs_dim, 128)
            self.p1 = nn.Linear(128, 64)
            self.ph = nn.Linear(64, 10)
            self.v0 = nn.Linear(obs_dim, 128)
            self.v1 = nn.Linear(128, 64)
            self.vh = nn.Linear(64, 1)
            with torch.no_grad():
                self.p0.weight.copy_(torch.tensor(np.asarray(p["pi"]["dense0"]["w"]).T))
                self.p0.bias.copy_(torch.tensor(np.asarray(p["pi"]["dense0"]["b"])))
                self.p1.weight.copy_(torch.tensor(np.asarray(p["pi"]["dense1"]["w"]).T))
                self.p1.bias.copy_(torch.tensor(np.asarray(p["pi"]["dense1"]["b"])))
                self.ph.weight.copy_(torch.tensor(np.asarray(p["pi"]["head"]["w"]).T))
                self.ph.bias.copy_(torch.tensor(np.asarray(p["pi"]["head"]["b"])))
                self.v0.weight.copy_(torch.tensor(np.asarray(p["vf"]["dense0"]["w"]).T))
                self.v0.bias.copy_(torch.tensor(np.asarray(p["vf"]["dense0"]["b"])))
                self.v1.weight.copy_(torch.tensor(np.asarray(p["vf"]["dense1"]["w"]).T))
                self.v1.bias.copy_(torch.tensor(np.asarray(p["vf"]["dense1"]["b"])))
                self.vh.weight.copy_(torch.tensor(np.asarray(p["vf"]["head"]["w"]).T))
                self.vh.bias.copy_(torch.tensor(np.asarray(p["vf"]["head"]["b"])))

        def forward(self, x):
            h = torch.tanh(self.p0(x))
            h = torch.tanh(self.p1(h))
            logits = self.ph(h)
            hv = torch.tanh(self.v0(x))
            hv = torch.tanh(self.v1(hv))
            return logits, self.vh(hv).squeeze(-1)

    net = Net(params)
    obs_t = torch.tensor(obs)
    act_t = torch.tensor(actions)
    old_logp_t = torch.tensor(old_logp)
    ret_t = torch.tensor(returns)
    adv_t = torch.tensor(adv)

    logits, v_pred = net(obs_t)
    logp = torch.zeros(B)
    ent = torch.zeros(B)
    for k, n_dim in enumerate((3, 3, 4)):
        dist = Categorical(logits=logits[:, k * 3 if k < 2 else 6: (k * 3 if k < 2 else 6) + n_dim])
        logp = logp + dist.log_prob(act_t[:, k])
        ent = ent + dist.entropy()
    ratio = torch.exp(logp - old_logp_t)
    adv_n = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    pg_loss = -torch.min(adv_n * ratio,
                         adv_n * torch.clamp(ratio, 0.9, 1.1)).mean()
    vf_loss = F.mse_loss(ret_t, v_pred)
    ent_loss = -ent.mean()
    loss = pg_loss + 0.03 * ent_loss + 0.5 * vf_loss

    net.zero_grad()
    loss.backward()   # gradients compared below BEFORE any in-place clipping

    # ── Part 1: loss + gradients vs torch (rtol absorbs f32 reduction-order
    # noise between torch-CPU and JAX; ~6e-4 relative observed, D39).  Any
    # loss-formula / entropy / advantage-normalisation bug shows up at O(1).
    pairs = [
        (("pi", "dense0", "w"), lambda: net.p0.weight.grad.numpy().T),
        (("pi", "dense0", "b"), lambda: net.p0.bias.grad.numpy()),
        (("pi", "dense1", "w"), lambda: net.p1.weight.grad.numpy().T),
        (("pi", "dense1", "b"), lambda: net.p1.bias.grad.numpy()),
        (("pi", "head", "w"), lambda: net.ph.weight.grad.numpy().T),
        (("pi", "head", "b"), lambda: net.ph.bias.grad.numpy()),
        (("vf", "dense0", "w"), lambda: net.v0.weight.grad.numpy().T),
        (("vf", "dense0", "b"), lambda: net.v0.bias.grad.numpy()),
        (("vf", "dense1", "w"), lambda: net.v1.weight.grad.numpy().T),
        (("vf", "dense1", "b"), lambda: net.v1.bias.grad.numpy()),
        (("vf", "head", "w"), lambda: net.vh.weight.grad.numpy().T),
        (("vf", "head", "b"), lambda: net.vh.bias.grad.numpy()),
    ]
    for (net_key, layer, name), get_ref in pairs:
        g_jax = np.asarray(grads[net_key][layer][name])
        ref = get_ref()
        scale = float(np.abs(ref).max()) if ref.size else 1.0
        np.testing.assert_allclose(
            g_jax, ref, rtol=1e-2, atol=max(1e-7, 2e-3 * scale),
            err_msg=f"grad mismatch: {net_key}/{layer}/{name}")

    # ── Part 2: optimizer wiring (D38) vs an independent transcription of
    # torch.optim.Adam(weight_decay=w) semantics — clip_grad_norm_(0.5) then
    # coupled L2 inside step(), eps=1e-8.  The SAME gradients feed both sides,
    # so only the optimizer mapping is under test.  (Comparing inside-jit grads
    # would add XLA-fusion noise; the first Adam step is sign-like and amplifies
    # any grad perturbation of near-zero elements — hence grads stay shared.)
    def adam_wd_numpy(p, g, m, v, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        total_norm = np.sqrt(sum(float((gi ** 2).sum()) for gi in g))
        coef = 0.5 / (total_norm + 1e-6)
        g = [gi * coef if coef < 1.0 else gi for gi in g]
        out = []
        for pi_, gi, mi, vi in zip(p, g, m, v):
            gi = gi + wd * pi_
            mi = beta1 * mi + (1 - beta1) * gi
            vi = beta2 * vi + (1 - beta2) * gi * gi
            m_hat = mi / (1 - beta1)
            v_hat = vi / (1 - beta2)
            out.append(pi_ - lr * m_hat / (np.sqrt(v_hat) + eps))
        return out

    flat_old, flat_grad = [], []
    for net_key, layer, name in [(n, l, nm) for (n, l, nm), _ in pairs]:
        flat_old.append(np.asarray(params[net_key][layer][name], dtype=np.float64))
        flat_grad.append(np.asarray(grads[net_key][layer][name], dtype=np.float64))
    zeros = [np.zeros_like(p) for p in flat_old]
    ref_new = adam_wd_numpy(flat_old, flat_grad, zeros, zeros, lr)

    opt = make_optimizer(lr, wd)
    opt_state = opt.init(params)
    updates, _ = opt.update(grads, opt_state, params)
    opt_new = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)

    for ((net_key, layer, name), _), ref in zip(pairs, ref_new):
        got = np.asarray(opt_new[net_key][layer][name], dtype=np.float64)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-9,
                                   err_msg=f"optimizer mapping: {net_key}/{layer}/{name}")

    # ── Part 3: the jitted update pipeline runs and moves the weights ────
    update_fn = make_update_fn(num_minibatches=1, n_epochs=1, clip_range=0.1,
                               ent_coef=0.03, vf_coef=0.5, target_kl=None,
                               weight_decay=wd)
    new_params, _opt_state, metrics = update_fn(
        params, opt.init(params), traj, jax.random.PRNGKey(3),
        jnp.asarray(lr, dtype=jnp.float32))
    assert float(metrics["n_updates"]) == 1.0
    assert np.isfinite(float(metrics["pg_loss"]))
    changed = any(
        not np.array_equal(np.asarray(new_params[n][l][nm]), np.asarray(params[n][l][nm]))
        for (n, l, nm), _ in pairs)
    assert changed, "update left all weights unchanged"


# ── RunningMeanStd vs SB3 ───────────────────────────────────────────────────


def test_running_moments_match_sb3():
    from stable_baselines3.common.running_mean_std import RunningMeanStd

    rng = np.random.default_rng(4)
    stream = rng.normal(size=(7, 5)) * 3.0 + 2.0

    ref = RunningMeanStd(shape=(5,))
    norm = init_norm_state(5)
    for row in stream:
        ref.update(row[None, :])
        norm = jax_ppo.update_obs_rms(norm, jnp.asarray(row[None, :]))

    np.testing.assert_allclose(np.asarray(norm.obs_mean), ref.mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(norm.obs_var), ref.var, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(norm.obs_count), ref.count, rtol=1e-12, atol=1e-12)


# ── VecNormalize stream vs SB3 ──────────────────────────────────────────────


def test_vecnormalize_stream_matches_sb3():
    """The full step semantics (update-then-normalise, discounted returns,
    reward clipping) against SB3's VecNormalize on the same scripted stream."""
    gym = pytest.importorskip("gymnasium")
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    obs_dim, n_steps = 6, 40
    rng = np.random.default_rng(5)
    raw_obs = rng.normal(size=(n_steps + 1, obs_dim)).astype(np.float32)
    raw_rew = rng.normal(size=n_steps).astype(np.float32)
    dones = np.zeros(n_steps, dtype=bool)
    dones[[7, 23]] = True   # scripted episode boundaries

    class Scripted(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
            self.action_space = spaces.Discrete(2)
            self.pos = 0   # number of steps served; the stream never rewinds

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return raw_obs[self.pos].copy(), {}

        def step(self, action):
            done = bool(dones[self.pos])
            obs = raw_obs[self.pos + 1].copy()
            rew = float(raw_rew[self.pos])
            self.pos += 1
            return obs, rew, done, False, {}

    venv = VecNormalize(DummyVecEnv([lambda: Scripted()]), norm_obs=True,
                        norm_reward=True, clip_obs=10.0, clip_reward=10.0,
                        gamma=0.99, training=True)
    ref_obs, ref_rew = [], []
    obs = venv.reset()
    ref_obs.append(np.asarray(obs[0], dtype=np.float32))
    for t in range(n_steps):
        obs, rew, _done, _info = venv.step(np.array([0]))
        ref_obs.append(np.asarray(obs[0], dtype=np.float32))
        ref_rew.append(float(rew[0]))

    # ── same stream through the functional port ─────────────────────────
    norm = init_norm_state(obs_dim)
    returns = jnp.zeros((1,), dtype=jnp.float64)
    norm = jax_ppo.update_obs_rms(norm, jnp.asarray(raw_obs[0][None, :]))
    got_obs = [np.asarray(normalize_obs(jnp.asarray(raw_obs[0][None, :]), norm)[0],
                          dtype=np.float32)]
    got_rew = []
    for t in range(n_steps):
        nxt = raw_obs[t + 1][None, :]
        norm = jax_ppo.update_obs_rms(norm, jnp.asarray(nxt))
        got_obs.append(np.asarray(normalize_obs(jnp.asarray(nxt), norm)[0], dtype=np.float32))
        returns = returns * 0.99 + jnp.asarray(raw_rew[t : t + 1], dtype=jnp.float64)
        norm = jax_ppo.update_ret_rms(norm, returns)
        got_rew.append(float(normalize_reward(jnp.asarray(raw_rew[t : t + 1]), norm)[0]))
        returns = jnp.where(jnp.asarray(dones[t : t + 1]), 0.0, returns)

    np.testing.assert_allclose(np.stack(got_obs), np.stack(ref_obs),
                               rtol=1e-6, atol=1e-6, err_msg="obs stream")
    np.testing.assert_allclose(np.asarray(got_rew), np.asarray(ref_rew),
                               rtol=1e-6, atol=1e-6, err_msg="reward stream")


# ── end-to-end smoke (D40/D42 artifact contract) ────────────────────────────


def test_train_smoke(tmp_path):
    from scenarios import build_frame, make_m1

    m1 = make_m1(n_days=45, seed=7)
    feat, cols = build_frame(m1)
    n = len(feat)
    train_feat = feat.iloc[: n - 250]
    val_feat = feat.iloc[n - 250:]
    datasets = (m1, cols, train_feat, val_feat, val_feat)

    state, rows = jax_ppo.train_jax(
        total_timesteps=512, seed=1, out_dir=str(tmp_path),
        train_episode_steps=64, eval_freq=256,
        datasets=datasets, num_envs=8, num_steps=32, num_minibatches=4,
        log_fn=lambda *_a, **_k: None,
    )

    run_info = json.loads((tmp_path / "run_info.json").read_text())
    assert run_info["framework"] == "jax"
    for key in ("slug", "model_path", "vecnorm_path", "decision_timeframe",
                "sl_atr_multipliers", "tp_r_multipliers", "total_timesteps",
                "train_episode_steps", "seed", "dd_penalty", "risk_fraction",
                "spread_price"):
        assert key in run_info, key
    assert Path(run_info["model_path"]).exists()

    csv_path = tmp_path / "eval_logs" / "consistency_evals.csv"
    assert csv_path.exists()
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["timesteps", "train_eval_r", "val_r", "train_dd_pct",
                                "val_dd_pct", "q_train", "q_val", "score", "gap",
                                "eligible"]
    assert len(df) >= 1

    # Checkpoint round-trip preserves params and norm bit-for-bit.
    params, norm = jax_ppo.load_checkpoint(run_info["model_path"])
    np.testing.assert_array_equal(
        np.asarray(params["pi"]["head"]["w"]), np.asarray(state.params["pi"]["head"]["w"]))
    np.testing.assert_array_equal(np.asarray(norm.obs_mean), np.asarray(state.norm.obs_mean))

    # Best checkpoint (when eligible) must carry the norm snapshot it was
    # selected under (ARTIFACTS.md invariant 1).
    best = tmp_path / "best_model" / "best_model.npz"
    if best.exists():
        assert (tmp_path / "best_model" / "best_model_norm.npz").exists()
        _b_params, b_norm = jax_ppo.load_checkpoint(best)
        b_snap = np.load(tmp_path / "best_model" / "best_model_norm.npz")
        for field in NormState._fields:
            np.testing.assert_array_equal(np.asarray(getattr(b_norm, field)),
                                          b_snap[field], err_msg=field)
