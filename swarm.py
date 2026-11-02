"""Swarm layer (Track B5, SCALING_PLAN.md B5) — seeds, batched gate evals,
population sweeps.

* **L1 — batch seeds** (``run_seed_sweep``): N single-seed trains on the same
  split, aggregated into a mean±std report.  Sequential by design (D53): the
  GPU is the bottleneck, so concurrent single-seed processes would time-share
  it anyway; determinism and simple VRAM use win on one device.
* **L2 — batched gate evals** (``batched_gate_eval``): score many windows on
  one checkpoint in a single jitted call via padded batches + per-window
  length masks (windows differ in T; padding rows are inert).
* **L3 — population sweeps** (``run_evosax_sweep``): evosax SimpleGA/Open_ES
  over PPO hyperparameters, fitness = short-train validation score
  (D54 — wrapper-style PBT, not weight-space ES; multi-device sharding is
  available in evosax but untested on this single-GPU box).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import jax_env
import jax_ppo
from jax_ppo import JaxModel, NormState


# ── L2: batched window scoring in one jitted call ───────────────────────────


class WindowBatch(NamedTuple):
    """Padded batch of per-window EnvParams + true lengths (D56)."""
    params: jax_env.EnvParams     # batched: leading dim B, rows padded to T_max
    lengths: Any                  # i32 [B] true decision-bar counts


def make_window_batch(env_params_list: list[jax_env.EnvParams]) -> WindowBatch:
    """Pad per-window EnvParams to a uniform [B, T_max, …] batch.

    Padding rows are inert by construction: win_len=0 (no touches), high −inf /
    low +inf, close/feat/atr zero — and the length mask freezes each window's
    state past its true end so padded rows are never stepped.
    """
    t_max = max(p.close.shape[0] for p in env_params_list)
    w_max = max(p.high_pad.shape[1] for p in env_params_list)
    lengths = jnp.asarray([p.close.shape[0] for p in env_params_list], dtype=jnp.int32)

    t_fields = ("feat", "close", "atr", "decision_ts",
                "high_pad", "low_pad", "ts_pad", "win_len")
    batched = {}
    for field in jax_env.EnvParams._fields:
        rows = [getattr(p, field) for p in env_params_list]
        if field not in t_fields:
            # Scalars and bracket vectors: just stack to leading batch dim.
            batched[field] = jnp.stack(rows)
            continue
        pad_value = {"high_pad": -jnp.inf, "low_pad": jnp.inf,
                     "ts_pad": 0, "win_len": 0}.get(field, 0.0)
        padded = []
        for x in rows:
            pad_width = [(0, t_max - x.shape[0])]
            if field in ("high_pad", "low_pad", "ts_pad"):   # (T, W): pad both axes
                pad_width.append((0, w_max - x.shape[1]))
            else:
                pad_width += [(0, 0)] * (x.ndim - 1)
            padded.append(jnp.pad(x, pad_width, constant_values=pad_value))
        batched[field] = jnp.stack(padded)
    return WindowBatch(params=jax_env.EnvParams(**batched), lengths=lengths)


def make_batched_eval_fn(num_envs: int):
    """Jitted ``eval(params, norm, batch)`` — one call scores ``num_envs``
    windows.  Returns per-step arrays [B, T_max-2] with inactive steps zeroed.
    """
    def eval_batched(params, norm: NormState, batch: WindowBatch):
        env_params = batch.params
        length_max = env_params.close.shape[1] - 2

        def init_one(p):
            state = jax_env.init_state(p, 0)
            return state, jax_ppo.normalize_obs(jax_env.observation(state, p), norm)

        states, obs_norm = jax.vmap(init_one)(env_params)
        step_idx = jnp.arange(length_max)

        def body(carry, t):
            states, obs_norm = carry
            active = t < (batch.lengths - 2)          # [B]
            actions = jax_ppo.deterministic_actions(
                jax_ppo.policy_logits(params, obs_norm))
            obs_next, new_states, rew, term, trunc, info = jax.vmap(
                jax_env.step, in_axes=(0, 0, 0))(
                states, actions, env_params)
            obs_next = jnp.where(active[:, None],
                                 jnp.where((term | trunc)[:, None],
                                           jnp.zeros_like(obs_next), obs_next),
                                 jnp.zeros_like(obs_next))
            new_states = jax.tree_util.tree_map(
                lambda fresh, old: jnp.where(active.reshape((-1,) + (1,) * (fresh.ndim - 1)),
                                             fresh, old),
                new_states, states)
            obs_norm = jax_ppo.normalize_obs(obs_next, norm)
            rew = jnp.where(active, rew, 0.0)
            closed = jnp.where(active[:, None], info.closed, False)
            # Inactive windows must hold their frozen state values — info was
            # produced by a garbage step into the padding rows.
            trans = {
                "reward": rew,
                "equity": jnp.where(active, info.equity, states.equity),
                "realized": jnp.where(active, info.realized, states.realized),
                "direction": jnp.where(active, info.direction, states.direction),
                "closed": closed,
                "reason": info.reason, "trade_direction": info.trade_direction,
                "entry_ts": info.entry_ts, "exit_ts": info.exit_ts,
                "entry_price": info.entry_price, "exit_price": info.exit_price,
                "sl": info.sl, "tp": info.tp, "sl_atr_mult": info.sl_atr_mult,
                "tp_r": info.tp_r, "units": info.units, "pnl": info.pnl,
                "r_mult": info.r_mult, "bars_in_trade": info.bars_in_trade,
            }
            return (new_states, obs_norm), trans

        _carry, out = jax.lax.scan(body, (states, obs_norm), step_idx)
        return out

    return jax.jit(eval_batched)


def batched_gate_eval(handle: JaxModel, windows: list[jax_env.EnvParams],
                      dd_penalty: float = 1.0) -> list[dict]:
    """Score every window in ONE jitted call (B5 level 2).

    Returns per-window dicts with reward, max_dd_pct, n_trades and the
    drawdown-penalised ``q = reward − dd_penalty·max_dd_pct`` leg score.
    """
    batch = make_window_batch(windows)
    b = len(windows)
    out = make_batched_eval_fn(b)(handle.params, handle.norm, batch)

    rewards = np.asarray(out["reward"])      # [steps, B]
    equity = np.asarray(out["equity"])       # [steps, B]
    closed = np.asarray(out["closed"])       # [steps, B, 2]
    lengths = np.asarray(batch.lengths) - 2
    rows = []
    for i in range(b):
        n = int(lengths[i])
        eq = equity[:n, i]
        peak = np.maximum.accumulate(eq) if n else eq
        max_dd_pct = float(np.abs((eq / peak - 1.0).min()) * 100.0) if n else 0.0
        reward = float(rewards[:n, i].sum())
        n_trades = int(closed[:n, i, :].sum())
        rows.append(dict(reward=reward, max_dd_pct=max_dd_pct, n_trades=n_trades,
                         q=reward - dd_penalty * max_dd_pct))
    return rows


# ── L1: multi-seed sweep ────────────────────────────────────────────────────


def run_seed_sweep(seeds: list[int], datasets: tuple, out_dir: str | Path,
                   total_timesteps: int = 100_000, dd_penalty: float = 1.0,
                   **train_kwargs) -> pd.DataFrame:
    """Train once per seed on the same split; aggregate val metrics (D53).

    Each seed gets its own subdirectory (ARTIFACTS layout); returns a DataFrame
    with one row per seed plus a printed mean±std summary — the direct fix for
    ANALYSIS.md §7's single-seed-42 weakness.
    """
    from jax_ppo import rollout_on_split, train_jax

    out_path = Path(out_dir)
    m1, feature_cols, _tr, val_feat, _te = datasets
    rows = []
    for seed in seeds:
        seed_dir = out_path / f"seed_{seed}"
        state, _ = train_jax(
            total_timesteps=total_timesteps, seed=seed, out_dir=str(seed_dir),
            datasets=datasets, dd_penalty=dd_penalty,
            log_fn=lambda *_a, **_k: None, **train_kwargs)
        handle = JaxModel(params=state.params, norm=state.norm)
        _eq, trades, rep = rollout_on_split(handle, m1, feature_cols, val_feat)
        rows.append({
            "seed": seed,
            "val_return_pct": rep.get("total_return_pct"),
            "val_profit_factor": rep.get("profit_factor"),
            "val_sharpe": rep.get("sharpe_like"),
            "val_n_trades": rep.get("n_trades"),
            "val_max_dd_pct": rep.get("max_drawdown_pct"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path / "seed_sweep.csv", index=False)
    print(df.to_string(index=False))
    for col in ("val_return_pct", "val_profit_factor", "val_sharpe"):
        print(f"  {col}: mean={df[col].mean():+.3f}  std={df[col].std():.3f}")
    return df


# ── L3: evosax population sweep over PPO hyperparameters ────────────────────

# Search space: (learning_rate, ent_coef, clip_range) normalised to [0,1]³.
HP_LO = jnp.asarray([1e-6, 0.0, 0.02])
HP_HI = jnp.asarray([3e-4, 0.15, 0.30])
HP_NAMES = ("learning_rate", "ent_coef", "clip_range")


def _decode_hp(x):
    x = jnp.clip(x, 0.0, 1.0)
    hp = HP_LO + x * (HP_HI - HP_LO)
    return {name: float(hp[i]) for i, name in enumerate(HP_NAMES)}


def run_evosax_sweep(datasets: tuple, out_dir: str | Path,
                     popsize: int = 8, n_generations: int = 4,
                     seed: int = 0, strategy: str = "SimpleGA",
                     total_timesteps: int = 20_000,
                     dd_penalty: float = 1.0) -> pd.DataFrame:
    """Wrapper-style population sweep (D54): evosax proposes hyperparameter
    vectors; each is scored by a short ``train_jax`` run + deterministic val
    rollout.  Minimises ``-q_val`` (drawdown-penalised val score).

    Not weight-space ES: this buys hyperparameter robustness now and reuses
    evosax's ask/tell machinery; policy-weight PBT remains open (B5).
    """
    import evosax.algorithms as ev
    from jax_ppo import rollout_on_split, train_jax

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    m1, feature_cols, _tr, val_feat, _te = datasets

    cls = {"SimpleGA": ev.SimpleGA, "Open_ES": ev.Open_ES,
           "PGPE": ev.PGPE, "CMA_ES": ev.CMA_ES}[strategy]
    template = jnp.zeros(3)
    alg = cls(population_size=popsize, solution=template)
    params = alg.default_params

    rng = jax.random.PRNGKey(seed)
    pop0 = jax.random.uniform(jax.random.fold_in(rng, 1), (popsize, 3))
    fit0 = jnp.full((popsize,), jnp.inf)
    state = alg.init(rng, pop0, fit0, params)

    history = []
    for gen in range(n_generations):
        rng, k1, k2 = jax.random.split(rng, 3)
        population, state = alg.ask(k1, state, params)
        fitness = np.empty((popsize,), dtype=np.float64)
        for i in range(popsize):
            hp = _decode_hp(population[i])
            t0 = time.time()
            tr_state, _ = train_jax(
                total_timesteps=total_timesteps, seed=seed, out_dir=str(out_path / "tmp"),
                datasets=datasets, dd_penalty=dd_penalty,
                log_fn=lambda *_a, **_k: None, **hp)
            handle = JaxModel(params=tr_state.params, norm=tr_state.norm)
            _eq, _trades, rep = rollout_on_split(handle, m1, feature_cols, val_feat)
            q = (float(rep.get("total_return_pct", 0.0))
                 - dd_penalty * abs(float(rep.get("max_drawdown_pct", 0.0))))
            fitness[i] = -q
            history.append(dict(generation=gen, member=i, q_val=q, **hp,
                                n_trades=rep.get("n_trades"),
                                seconds=round(time.time() - t0, 1)))
            print(f"  gen {gen} member {i}: q_val={q:+.2f} hp={hp}")
        state, metrics = alg.tell(k2, population, jnp.asarray(fitness), state, params)

    df = pd.DataFrame(history)
    df.to_csv(out_path / "evosax_sweep.csv", index=False)
    best_hp = _decode_hp(state.best_solution)
    summary = {"best_q_val": -float(state.best_fitness), "best_hp": best_hp,
               "strategy": strategy, "popsize": popsize,
               "n_generations": n_generations}
    (out_path / "evosax_best.json").write_text(json.dumps(summary, indent=2))
    print(f"best: q_val={summary['best_q_val']:+.2f}  hp={best_hp}")
    return df


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seeds", help="L1 multi-seed sweep")
    p_seed.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p_seed.add_argument("--steps", type=int, default=100_000)

    p_gate = sub.add_parser("gate-eval", help="L2 batched window scoring demo")
    p_gate.add_argument("--checkpoint", type=Path, required=True)
    p_gate.add_argument("--folds", type=int, default=3)

    p_ev = sub.add_parser("evosax", help="L3 population hp sweep")
    p_ev.add_argument("--strategy", default="SimpleGA",
                      choices=["SimpleGA", "Open_ES", "PGPE", "CMA_ES"])
    p_ev.add_argument("--popsize", type=int, default=8)
    p_ev.add_argument("--generations", type=int, default=4)
    p_ev.add_argument("--steps", type=int, default=20_000)

    args = ap.parse_args()

    if args.cmd == "seeds":
        import train_ppo as tp
        m1, cols, tr, va, te = tp.load_datasets()
        run_seed_sweep(args.seeds, (m1, cols, tr, va, te),
                       out_dir="models/seed_sweep",
                       total_timesteps=args.steps)
    elif args.cmd == "gate-eval":
        import train_ppo as tp
        from data_loader import make_sliding_folds
        from config import CFG
        handle = jax_ppo.load_model(args.checkpoint)
        m1, feat, cols = tp._load_decision_features()
        folds = make_sliding_folds(
            feat, train_years=CFG.sliding_train_years,
            val_months=CFG.sliding_val_months, test_months=CFG.sliding_test_months,
            step_months=CFG.sliding_step_months, embargo_bars=CFG.split_embargo_bars)[:args.folds]
        windows = [
            jax_ppo.build_env_params_cfg(te, tp._slice_m1_for_decision_window(m1, te),
                                         cols, None)
            for _tr, _va, te in folds]
        rows = batched_gate_eval(handle, windows)
        print(pd.DataFrame(rows).to_string())
    elif args.cmd == "evosax":
        import train_ppo as tp
        m1, cols, tr, va, te = tp.load_datasets()
        run_evosax_sweep((m1, cols, tr, va, te), out_dir="models/evosax_sweep",
                         popsize=args.popsize, n_generations=args.generations,
                         strategy=args.strategy, total_timesteps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
