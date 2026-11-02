"""Generate golden trace fixtures from the CURRENT environment implementation.

Run this BEFORE refactoring ``env_bracket.py`` (DECISIONS.md P1): the fixtures
freeze the present behaviour so the refactor can prove parity.  Regenerate
deliberately (and review the diff!) only when a behaviour change is intended.

Usage::

    python tests/make_golden.py            # write tests/golden/*
    python tests/make_golden.py --check    # compare against existing fixtures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import SCENARIOS, TRACE_KEYS, run_trace  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden"


def trace_to_frame(trade_log: pd.DataFrame) -> pd.DataFrame:
    return trade_log


def write_fixture(name: str, trace: dict, trade_log: pd.DataFrame) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    arrays = {k: trace[k] for k in list(TRACE_KEYS) + [
        "start_i", "hist_equity", "hist_realized", "hist_close",
        "hist_reward", "hist_position", "hist_time_ns"]}
    np.savez_compressed(GOLDEN / f"{name}.npz", **arrays)
    trade_log.to_csv(GOLDEN / f"{name}_trades.csv", index=False)


def load_fixture(name: str):
    data = np.load(GOLDEN / f"{name}.npz")
    trace = {k: data[k] for k in data.files}
    try:
        trades = pd.read_csv(GOLDEN / f"{name}_trades.csv")
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame()   # scenario with zero trades
    return trace, trades


def compare(name: str, new_trace: dict, new_trades: pd.DataFrame) -> list[str]:
    """Return a list of human-readable mismatches (empty == parity)."""
    problems: list[str] = []
    try:
        old_trace, old_trades = load_fixture(name)
    except FileNotFoundError:
        return [f"{name}: fixture missing (run without --check to generate)"]

    for key in list(TRACE_KEYS) + ["hist_equity", "hist_realized", "hist_close",
                             "hist_reward", "hist_position", "hist_time_ns",
                             "start_i"]:
        a, b = old_trace[key], new_trace[key]
        if a.shape != b.shape:
            problems.append(f"{name}.{key}: shape {a.shape} != {b.shape}")
            continue
        if a.dtype == np.float32:
            ok = np.array_equal(a, b)
        elif np.issubdtype(a.dtype, np.floating):
            ok = np.array_equal(a, b) or np.allclose(a, b, rtol=0, atol=1e-9)
        else:
            ok = np.array_equal(a, b)
        if not ok:
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64)) if a.size else 0
            problems.append(f"{name}.{key}: values differ (max abs diff={np.max(diff) if np.size(diff) else 0})")

    if len(old_trades) != len(new_trades):
        problems.append(f"{name}.trades: {len(old_trades)} rows != {len(new_trades)}")
    elif len(old_trades):
        for col in old_trades.columns:
            a, b = old_trades[col].to_numpy(), new_trades[col].to_numpy()
            if np.issubdtype(a.dtype, np.floating):
                if not (np.array_equal(a, b) or np.allclose(a, b, rtol=0, atol=1e-9)):
                    problems.append(f"{name}.trades.{col}: values differ")
            else:
                if not np.array_equal(a.astype(str), b.astype(str)):
                    problems.append(f"{name}.trades.{col}: values differ")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare current behaviour against fixtures")
    args = ap.parse_args()

    all_problems: list[str] = []
    for sc in SCENARIOS:
        trace, trades = run_trace(sc)
        if args.check:
            probs = compare(sc.name, trace, trades)
            all_problems.extend(probs)
            status = "OK" if not probs else "MISMATCH"
        else:
            write_fixture(sc.name, trace, trades)
            status = f"wrote ({len(trades)} trades, {len(trace['reward'])} steps)"
        print(f"  {sc.name:24s} {status}")

    if args.check:
        if all_problems:
            print("\nPARITY FAILED:")
            for p in all_problems:
                print(f"  - {p}")
            return 1
        print("\nAll fixtures match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
