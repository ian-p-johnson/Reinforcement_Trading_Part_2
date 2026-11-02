"""Runnable benchmark suite for the RL engine (shared asset, SCALING_PLAN.md).

Reproduces the measurements in ANALYSIS.md §4 so before/after numbers stay
comparable across refactors and across tracks.  Usage::

    python benchmarks.py data            # data-ingress timings
    python benchmarks.py env             # single-process env stepping
    python benchmarks.py vec             # SubprocVecEnv + VecNormalize
    python benchmarks.py learn           # end-to-end PPO.learn variants
    python benchmarks.py all             # everything (slow)

All timings run on synthetic/scenario data where possible so the suite needs
no external dataset; ``data`` uses the real CSV when present and falls back to
synthetic M1 otherwise (the result is flagged in the output).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

RESULTS: list[tuple[str, str, str]] = []


def record(group: str, name: str, detail: str) -> None:
    RESULTS.append((group, name, detail))
    print(f"[{group:5s}] {name:52s} {detail}", flush=True)


# ── data ingress ─────────────────────────────────────────────────────────────

def bench_data() -> None:
    from config import CFG
    from data_loader import load_mt_ohlcv_csv, resample_ohlcv, split_train_val_test
    from features import prepare_feature_frame

    real = Path(CFG.csv_path).exists()
    if real:
        t0 = time.perf_counter()
        m1 = load_mt_ohlcv_csv(CFG.csv_path, time_col=CFG.time_col,
                               source_tz=CFG.source_tz,
                               timestamp_is_bar_open=CFG.timestamp_is_bar_open,
                               bar_duration=CFG.pandas_execution_tf)
        t1 = time.perf_counter()
        record("data", "load_mt_ohlcv_csv (real CSV)",
               f"{t1 - t0:.2f}s  rows={len(m1):,}")
    else:
        from scenarios import make_m1
        t0 = time.perf_counter()
        m1 = make_m1(n_days=400, seed=0)
        t1 = time.perf_counter()
        record("data", "make_m1 (synthetic; real CSV absent)",
               f"{t1 - t0:.2f}s  rows={len(m1):,}")

    t0 = time.perf_counter()
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, cols = prepare_feature_frame(decision, CFG.warmup_bars,
                                       CFG.atr_period, CFG.rsi_period)
    t1 = time.perf_counter()
    record("data", "resample + prepare_feature_frame",
           f"{t1 - t0:.2f}s  H1 bars={len(feat):,}")

    tr, _va, _te = split_train_val_test(feat, CFG.train_frac, CFG.val_frac,
                                        CFG.split_embargo_bars)
    t0 = time.perf_counter()
    sl = m1.loc[(m1.index > tr.index.min()) & (m1.index <= tr.index.max())].copy()
    t1 = time.perf_counter()
    record("data", "M1 slice copy", f"{t1 - t0:.2f}s  rows={len(sl):,}")

    try:
        from data_cache import load_m1_cached, load_feature_frame_cached
        t0 = time.perf_counter()
        _m1b = load_m1_cached()
        t1 = time.perf_counter()
        record("data", "load_m1_cached (parquet)", f"{t1 - t0:.2f}s")
        t0 = time.perf_counter()
        _fb, _cb = load_feature_frame_cached()
        t1 = time.perf_counter()
        record("data", "load_feature_frame_cached (parquet)", f"{t1 - t0:.2f}s")
    except Exception as exc:  # cache optional in early runs
        record("data", "cache", f"unavailable ({type(exc).__name__}: {exc})")


# ── env stepping ─────────────────────────────────────────────────────────────

def _scenario_env():
    from scenarios import SCENARIOS
    sc = {s.name: s for s in SCENARIOS}["random_seed7"]
    return sc.make_env()


def bench_env(n: int = 8000) -> None:
    env = _scenario_env()
    rng = np.random.default_rng(0)

    def run(label, action_fn):
        env.reset(seed=0)
        t0 = time.perf_counter()
        k = 0
        while k < n:
            _o, _r, term, trunc, _i = env.step(action_fn())
            k += 1
            if term or trunc:
                env.reset()
        dt = time.perf_counter() - t0
        record("env", label, f"{n / dt:9.0f} steps/s  ({dt / n * 1e6:.1f} us/step)")

    run("single-proc env (random actions, incl. trades)",
        lambda: rng.integers([0, 0, 0], [3, 3, 4]))
    run("single-proc env (flat, no intrabar sim)", lambda: np.array([0, 1, 1]))
    run("single-proc env (long hold, in-trade intrabar)",
        lambda: np.array([1, 1, 1]))

    env.reset(seed=0)
    t = []
    for _ in range(20000):
        ta = time.perf_counter()
        env._observation()
        t.append(time.perf_counter() - ta)
    record("env", "_observation() alone", f"{np.mean(t) * 1e6:.1f} us mean")

    t = []
    for _ in range(20000):
        ta = time.perf_counter()
        env._current_row()
        t.append(time.perf_counter() - ta)
    record("env", "_current_row() compat shim", f"{np.mean(t) * 1e6:.1f} us mean")


# ── vectorised env stack ─────────────────────────────────────────────────────

def bench_vec(n_envs: int = 4, n: int = 20000) -> None:
    from functools import partial

    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from scenarios import SCENARIOS, m1_slice_for
    from train_ppo import _spawn_train_env

    sc = {s.name: s for s in SCENARIOS}["random_seed7"]
    m1_sl = m1_slice_for(sc.m1, sc.decision_df)
    fns = [partial(_spawn_train_env, sc.decision_df, m1_sl, sc.feature_cols,
                   2048) for _ in range(n_envs)]
    sub = VecNormalize(SubprocVecEnv(fns, start_method="forkserver"),
                       norm_obs=True, norm_reward=True)
    obs = sub.reset()
    rng = np.random.default_rng(0)
    steps = 0
    t0 = time.perf_counter()
    while steps < n:
        a = np.array([rng.integers([0, 0, 0], [3, 3, 4]) for _ in range(n_envs)])
        obs, _r, _d, _i = sub.step(a)
        steps += n_envs
    dt = time.perf_counter() - t0
    record("vec", f"SubprocVecEnv x{n_envs} + VecNormalize (random actions)",
           f"{steps / dt:9.0f} steps/s")
    sub.close()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ── end-to-end PPO ───────────────────────────────────────────────────────────

def bench_learn(n_envs: int = 4, timesteps: int = 100_000) -> None:
    from functools import partial

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from config import CFG
    from scenarios import SCENARIOS, m1_slice_for
    from train_ppo import _spawn_train_env

    sc = {s.name: s for s in SCENARIOS}["random_seed7"]
    m1_sl = m1_slice_for(sc.m1, sc.decision_df)

    def run(label, device, torch_threads):
        torch.set_num_threads(torch_threads)
        fns = [partial(_spawn_train_env, sc.decision_df, m1_sl,
                       sc.feature_cols, 2048) for _ in range(n_envs)]
        sub = VecNormalize(SubprocVecEnv(fns, start_method="forkserver"),
                           norm_obs=True, norm_reward=True)
        model = PPO("MlpPolicy", sub, device=device, n_steps=2048,
                    batch_size=min(1024, 2048 * n_envs), n_epochs=5,
                    policy_kwargs={"net_arch": list(CFG.ppo_net_arch)},
                    verbose=0, seed=1)
        t0 = time.perf_counter()
        model.learn(total_timesteps=timesteps)
        dt = time.perf_counter() - t0
        record("learn", f"{label}  (n_envs={n_envs}, {timesteps // 1000}k steps)",
               f"{timesteps / dt:9.0f} steps/s  ({dt:.1f}s)")
        sub.close()

    if torch.cuda.is_available():
        run("PPO.learn cuda", "cuda", 20)
    run("PPO.learn cpu (torch threads=4)", "cpu", 4)
    run("PPO.learn cpu (torch threads=20 = default)", "cpu", 20)


def bench_learn_sweep(n_envs_list=(4, 8, 16), timesteps: int = 100_000) -> None:
    """A3: end-to-end throughput vs vectorization width."""
    import torch
    from functools import partial
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from config import CFG
    from scenarios import SCENARIOS, m1_slice_for
    from train_ppo import _spawn_train_env

    sc = {s.name: s for s in SCENARIOS}["random_seed7"]
    m1_sl = m1_slice_for(sc.m1, sc.decision_df)

    for device, threads in (("cuda", 20), ("cpu", 4)):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        torch.set_num_threads(threads)
        for n_envs in n_envs_list:
            fns = [partial(_spawn_train_env, sc.decision_df, m1_sl,
                           sc.feature_cols, 2048) for _ in range(n_envs)]
            sub = VecNormalize(SubprocVecEnv(fns, start_method="forkserver"),
                               norm_obs=True, norm_reward=True)
            buf = 2048 * n_envs
            model = PPO("MlpPolicy", sub, device=device, n_steps=2048,
                        batch_size=min(2048, buf), n_epochs=5,
                        policy_kwargs={"net_arch": list(CFG.ppo_net_arch)},
                        verbose=0, seed=1)
            t0 = time.perf_counter()
            model.learn(total_timesteps=timesteps)
            dt = time.perf_counter() - t0
            record("sweep", f"{device:4s} thr={threads:2d} n_envs={n_envs:2d}",
                   f"{timesteps / dt:9.0f} steps/s  ({dt:.1f}s)")
            sub.close()


GROUPS = {
    "data": bench_data,
    "env": bench_env,
    "vec": bench_vec,
    "learn": bench_learn,
    "sweep": bench_learn_sweep,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("groups", nargs="*", default=["env"],
                    choices=sorted(GROUPS) + ["all"])
    args = ap.parse_args()
    groups = list(GROUPS) if "all" in args.groups else args.groups
    for g in groups:
        print(f"\n=== {g} ===", flush=True)
        GROUPS[g]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
