"""Track B5 correctness tests for the swarm layer and benchmark harness.

Everything here is *non-timing*: batched-eval equivalence, seed-sweep
aggregation, evosax ask/tell wiring and the benchmark harness at ``--quick``
sizes (whose numbers are never quotable).  The recorded throughput runs are
scheduled separately on a quiet machine (SCALING_PLAN.md B5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import conftest  # noqa: F401

jax = pytest.importorskip("jax")
jnp = jax.numpy

import jax_env  # noqa: E402
import jax_ppo  # noqa: E402
import swarm  # noqa: E402


def _synthetic_datasets(n_days=45):
    from scenarios import build_frame, make_m1
    m1 = make_m1(n_days=n_days, seed=7)
    feat, cols = build_frame(m1)
    n = len(feat)
    tr = feat.iloc[: n - 400]
    va = feat.iloc[n - 400: n - 200]
    te = feat.iloc[n - 200:]
    return m1, cols, tr, va, te


def _window_params(m1, cols, df):
    return jax_ppo.build_env_params_cfg(
        df, jax_ppo.slice_m1_for_decision_window(m1, df), cols, episode_steps=None)


# ── L2: batched gate eval ≡ per-window loop ─────────────────────────────────


def test_window_batch_padding_is_inert():
    m1, cols, tr, va, te = _synthetic_datasets()
    windows = [_window_params(m1, cols, df) for df in (tr, va, te)]
    batch = swarm.make_window_batch(windows)
    t_max = max(w.close.shape[0] for w in windows)
    assert batch.params.close.shape == (3, t_max)
    assert batch.params.feat.shape[0] == 3
    assert np.asarray(batch.lengths).tolist() == [w.close.shape[0] for w in windows]
    # Padding rows: no touches possible.
    win_len = np.asarray(batch.params.win_len)
    for i, w in enumerate(windows):
        t = w.close.shape[0]
        assert win_len[i, t:].sum() == 0
        assert np.isneginf(np.asarray(batch.params.high_pad)[i, t:]).all()
        assert np.isposinf(np.asarray(batch.params.low_pad)[i, t:]).all()


def test_batched_gate_eval_matches_loop():
    m1, cols, tr, va, te = _synthetic_datasets()
    windows = [_window_params(m1, cols, df) for df in (tr.iloc[:120], va, te)]
    handle = jax_ppo.JaxModel(
        params=jax_ppo.init_params(jax.random.PRNGKey(0), len(cols) + 6),
        norm=jax_ppo.init_norm_state(len(cols) + 6))

    rows = swarm.batched_gate_eval(handle, windows, dd_penalty=1.0)

    for i, w in enumerate(windows):
        out = jax_ppo.eval_fn(handle.params, handle.norm, w)
        rew = float(np.asarray(out["reward"]).sum())
        eq = np.asarray(out["equity"], dtype=np.float64)
        peak = np.maximum.accumulate(eq)
        dd = float(np.abs((eq / peak - 1.0).min()) * 100.0)
        n_trades = int(np.asarray(out["closed"]).sum())
        assert rows[i]["n_trades"] == n_trades, f"window {i}"
        np.testing.assert_allclose(rows[i]["reward"], rew, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rows[i]["max_dd_pct"], dd, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(rows[i]["q"], rew - dd, rtol=1e-9, atol=1e-9)


# ── L1: seed sweep ──────────────────────────────────────────────────────────


def test_seed_sweep_smoke(tmp_path):
    m1, cols, tr, va, te = _synthetic_datasets()
    df = swarm.run_seed_sweep(
        seeds=[0, 1], datasets=(m1, cols, tr, va, te), out_dir=tmp_path,
        total_timesteps=512, num_envs=8, num_steps=32, num_minibatches=4,
    )
    assert len(df) == 2
    for col in ("seed", "val_return_pct", "val_profit_factor", "val_sharpe",
                "val_n_trades", "val_max_dd_pct"):
        assert col in df.columns, col
    assert (tmp_path / "seed_sweep.csv").exists()
    assert (tmp_path / "seed_0" / "run_info.json").exists()
    assert (tmp_path / "seed_1" / "run_info.json").exists()


# ── L3: evosax sweep ────────────────────────────────────────────────────────


def test_evosax_sweep_smoke(tmp_path):
    m1, cols, tr, va, te = _synthetic_datasets()
    df = swarm.run_evosax_sweep(
        (m1, cols, tr, va, te), out_dir=tmp_path,
        popsize=3, n_generations=2, seed=0, strategy="SimpleGA",
        total_timesteps=256,
    )
    assert len(df) == 6, "popsize x generations rows"
    for col in ("generation", "member", "q_val", *swarm.HP_NAMES):
        assert col in df.columns, col
    best = json.loads((tmp_path / "evosax_best.json").read_text())
    assert set(best) >= {"best_q_val", "best_hp", "strategy"}
    assert set(best["best_hp"]) == set(swarm.HP_NAMES)
    # HP decode stays inside the documented bounds.
    for name in swarm.HP_NAMES:
        lo = float(swarm.HP_LO[swarm.HP_NAMES.index(name)])
        hi = float(swarm.HP_HI[swarm.HP_NAMES.index(name)])
        assert lo <= best["best_hp"][name] <= hi, name


# ── benchmark harness wiring (quick sizes only) ─────────────────────────────


def test_benchmark_harness_wires():
    import jax_benchmarks as jb

    jb.RESULTS.clear()
    jb.bench_env(data="synthetic", quick=True)
    jb.bench_train(data="synthetic", quick=True)
    jb.bench_compile(data="synthetic", quick=True)
    jb.bench_transfer(data="synthetic", quick=True)
    jb.bench_vram(data="synthetic", quick=True)
    groups = {g for g, _n, _d in jb.RESULTS}
    assert groups == {"env", "train", "comp", "xfer", "vram"}, groups
    assert all(detail for _g, _n, detail in jb.RESULTS)
