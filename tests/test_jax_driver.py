"""Track B4 gate — sliding walk-forward driver on the JAX engine (SCALING_PLAN B4).

Runs the real driver end-to-end at toy budget on the real dataset into a
temporary out_dir (never ``models/``), covering:

  * sequential folds (``_run_sliding_fold`` engine branch),
  * parallel folds (``_fold_worker_main`` engine routing + XLA VRAM sharing),
  * artifact contract (``fold_result.json``, ``test_equity.csv``,
    ``test_trades.csv``, ``sliding_walk_forward_summary.csv``, stitched OOS,
    production-slot promotion + ``run_info.json`` framework key),
  * schema parity of engine outputs (SB3 vs JAX, per B4 "schema-identical
    outputs" — metrics legitimately differ across trainers, D48).

Skipped without the real dataset.  Fold count is reduced via a
``CFG.sliding_step_months`` override so the test stays minutes, not hours.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conftest  # noqa: F401

pytest.importorskip("jax")

HAS_REAL_DATA = Path("data/XAUUSD_1m_EET.csv").exists()
pytestmark = pytest.mark.skipif(not HAS_REAL_DATA, reason="real dataset not present")


@pytest.fixture()
def few_folds_cfg():
    """Reduce the fold count to ~3 and restore CFG afterwards."""
    from config import CFG
    old = CFG.sliding_step_months
    CFG.sliding_step_months = 84   # 7-year slide -> ~3 folds over 23y
    yield CFG
    CFG.sliding_step_months = old


def _driver(out_dir: str, engine: str, n_parallel: int, total_timesteps: int = 25_000):
    from train_ppo import train_sliding_walk_forward
    return train_sliding_walk_forward(
        total_timesteps=total_timesteps, seed=7, out_dir=out_dir,
        train_episode_steps=2048, target_evals_per_fold=20, dd_penalty=0.8,
        n_envs=1, device="cpu", n_parallel_folds=n_parallel, engine=engine,
    )


def _assert_artifact_contract(out_dir: Path, engine: str) -> dict:
    summary_path = out_dir / "sliding_walk_forward_summary.csv"
    assert summary_path.exists(), "sliding summary missing"
    summary = pd.read_csv(summary_path)
    expect_cols = ["fold", "train_start", "train_end", "test_start", "test_end",
                   "val_return_pct", "val_profit_factor", "test_return_pct",
                   "test_sharpe", "test_profit_factor", "test_win_rate_pct",
                   "test_max_dd_pct", "test_avg_r", "test_n_trades"]
    for col in expect_cols:
        assert col in summary.columns, f"summary column {col}"

    stitched = pd.read_csv(out_dir / "sliding_oos_equity.csv", index_col=0,
                           parse_dates=True)
    assert "equity" in stitched.columns and len(stitched) > 0, "stitched OOS empty"

    folds = sorted((out_dir / "sliding").glob("fold_*"))
    assert folds, "no fold directories"
    for fold_dir in folds:
        assert (fold_dir / "fold_result.json").exists(), f"{fold_dir}: fold_result"
        assert (fold_dir / "test_equity.csv").exists(), f"{fold_dir}: test_equity"
        assert (fold_dir / "test_trades.csv").exists(), f"{fold_dir}: test_trades"
        eq = pd.read_csv(fold_dir / "test_equity.csv", index_col=0, parse_dates=True)
        assert "equity" in eq.columns
        row = json.loads((fold_dir / "fold_result.json").read_text())
        for key in ("fold", "test_return_pct", "test_sharpe", "test_profit_factor",
                    "test_n_trades"):
            assert key in row, key
        ri = json.loads((fold_dir / "run_info.json").read_text())
        assert ri.get("framework", "sb3") == (
            "jax" if engine == "jax" else "sb3")

    # Production slot: promoted run_info + final checkpoint; best_model is
    # present iff the promoted fold produced an eligible checkpoint (the
    # do-nothing guard may legitimately find none at toy budget).
    prod_ri = out_dir / "run_info.json"
    assert prod_ri.exists(), "production run_info missing"
    prod = json.loads(prod_ri.read_text())
    assert "gate_passed" in prod and "promoted_from_fold" in prod
    assert Path(prod["model_path"]).exists(), "final checkpoint not promoted"
    fold_best = (out_dir / "sliding" / f"fold_{prod['promoted_from_fold']}"
                 / "best_model")
    if fold_best.exists():
        assert (out_dir / "best_model").exists(), "best_model not promoted"
        if engine == "jax":
            assert (out_dir / "best_model" / "best_model.npz").exists()
            assert (out_dir / "best_model" / "best_model_norm.npz").exists()
    if engine == "jax":
        assert prod.get("framework") == "jax"
        assert Path(prod["vecnorm_path"]).exists()
    assert (out_dir / "NO_DEPLOY.txt").exists() or prod["gate_passed"]

    # Consistency eval log schema (ARTIFACTS.md) wherever it exists.
    for csv_path in (out_dir / "sliding").glob("fold_*/eval_logs/consistency_evals.csv"):
        df = pd.read_csv(csv_path)
        assert list(df.columns) == [
            "timesteps", "train_eval_r", "val_r", "train_dd_pct", "val_dd_pct",
            "q_train", "q_val", "score", "gap", "eligible"], csv_path
    return summary


def test_sliding_driver_end_to_end_jax(tmp_path, few_folds_cfg):
    out_dir = tmp_path / "jax_run"
    summary = _driver(str(out_dir), engine="jax", n_parallel=1)
    assert len(summary) >= 2, "expected at least 2 folds at 7-year slide"
    _assert_artifact_contract(out_dir, "jax")


def test_sliding_driver_parallel_jax(tmp_path, few_folds_cfg):
    out_dir = tmp_path / "jax_par"
    summary = _driver(str(out_dir), engine="jax", n_parallel=2)
    assert len(summary) >= 2
    _assert_artifact_contract(out_dir, "jax")
    for fold_dir in (out_dir / "sliding").glob("fold_*"):
        assert (fold_dir / "fold_spec.json").exists()
        assert (fold_dir / "worker.log").exists()


def test_engine_output_schema_parity(tmp_path, few_folds_cfg):
    """B4 'schema-identical outputs': SB3 and JAX drivers must produce the
    same artifact files and column schemas (metrics may differ — D48)."""
    sb3_dir = tmp_path / "sb3_run"
    jax_dir = tmp_path / "jax_run"
    s_sb3 = _driver(str(sb3_dir), engine="sb3", n_parallel=1)
    s_jax = _driver(str(jax_dir), engine="jax", n_parallel=1)

    assert sorted(s_sb3.columns) == sorted(s_jax.columns), "summary schema drift"
    assert s_sb3["fold"].tolist() == s_jax["fold"].tolist()

    for name in ("sliding_oos_equity.csv", "run_info.json"):
        assert (sb3_dir / name).exists() and (jax_dir / name).exists(), name

    for fold_dir in sorted((sb3_dir / "sliding").glob("fold_*")):
        jax_fold = jax_dir / "sliding" / fold_dir.name
        for csv_name in ("test_equity.csv", "test_trades.csv"):
            a = pd.read_csv(fold_dir / csv_name)
            b = pd.read_csv(jax_fold / csv_name)
            assert sorted(a.columns) == sorted(b.columns), (
                f"{fold_dir.name}/{csv_name}: {sorted(a.columns)} != {sorted(b.columns)}")
        a_row = json.loads((fold_dir / "fold_result.json").read_text())
        b_row = json.loads((jax_fold / "fold_result.json").read_text())
        assert sorted(a_row.keys()) == sorted(b_row.keys()), fold_dir.name
