"""
view_results.py — visualise any slice of a saved simulation without re-running.

Reloads OHLCV data and recomputes indicators (seconds), then overlays the
trade log and equity curve that were saved to outputs/ by run_pipeline.py.

By default this viewer points at validation-only artifacts so the test split
can remain sealed during iteration. Pass explicit holdout files after running
final_holdout_eval.py.

CLI usage
---------
  python view_results.py                              # last 500 decision bars
  python view_results.py --bars 1000                 # last 1000 bars
  python view_results.py --start 2024-01-01          # from date to end
  python view_results.py --end   2024-06-01          # from start to date
  python view_results.py --start 2024-01-01 --end 2024-06-01
  python view_results.py --split train               # train split per CFG fracs
  python view_results.py --split val
  python view_results.py --split test
  python view_results.py --no-browser                # save HTML, don't auto-open

Python API
----------
  from view_results import view_slice
  view_slice(start="2024-06-01", end="2024-09-01")
  view_slice(split="test", bars=300)
  view_slice(bars=200, open_browser=False)
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import pandas as pd

from config import CFG
from data_loader import split_train_val_test
from evaluate import full_report
from visualize import (
    plot_candles_with_indicators,
    plot_equity_and_drawdown,
    plot_feature_correlation,
    plot_trades_on_chart,
    save_html,
)

_OUT = Path("outputs")
_VIEW = Path("outputs/view")


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_features() -> tuple[pd.DataFrame, list[str]]:
    """Reload M1, resample, compute features (parquet-cached; ~1 s warm)."""
    from data_cache import load_all
    _m1, _decision, feat, feature_cols = load_all()
    return feat, feature_cols


def _load_trades(path: Path = _OUT / "selected_val_trade_log.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run run_pipeline.py or train_ppo.py first."
        )
    trades = pd.read_csv(path)
    for col in ("entry_time", "exit_time"):
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], utc=True).dt.tz_convert(CFG.source_tz)
    return trades


def _load_equity(path: Path = _OUT / "selected_val_equity_curve.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run run_pipeline.py or train_ppo.py first."
        )
    eq = pd.read_csv(path, index_col=0)
    eq.index = pd.to_datetime(eq.index, utc=True).tz_convert(CFG.source_tz)
    eq.index.name = "time"
    return eq


def _split_bounds(
    feat: pd.DataFrame, split: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) timestamps for a named split."""
    train, val, test = split_train_val_test(
        feat,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        embargo_bars=CFG.split_embargo_bars,
    )
    mapping = {"train": train, "val": val, "test": test}
    if split not in mapping:
        raise ValueError(f"split must be one of {list(mapping)}, got '{split}'")
    df = mapping[split]
    return df.index.min(), df.index.max()


def _filter_feat(
    feat: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    bars: int | None,
    tz: str,
) -> pd.DataFrame:
    d = feat.copy()
    if start:
        d = d.loc[pd.Timestamp(start).tz_localize(tz, ambiguous="infer",
                                                   nonexistent="shift_forward"):]
    if end:
        d = d.loc[:pd.Timestamp(end).tz_localize(tz, ambiguous="infer",
                                                  nonexistent="shift_forward")]
    if bars and bars < len(d):
        d = d.tail(bars)
    return d


def _filter_equity(
    equity: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    d = equity.copy()
    if start:
        d = d.loc[start:]
    if end:
        d = d.loc[:end]
    return d


def _filter_trades(
    trades: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    d = trades.copy()
    if start:
        d = d[d["entry_time"] >= start]
    if end:
        d = d[d["entry_time"] <= end]
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def view_slice(
    start: str | None = None,
    end: str | None = None,
    bars: int | None = 500,
    split: str | None = None,
    trade_log: str | Path = _OUT / "selected_val_trade_log.csv",
    equity_csv: str | Path = _OUT / "selected_val_equity_curve.csv",
    out_dir: str | Path = _VIEW,
    open_browser: bool = True,
) -> Path:
    """Generate HTML charts for a time slice and optionally open them.

    Parameters
    ----------
    start, end : date strings like "2024-01-01"
    bars       : show last N decision bars (applied after start/end filtering)
    split      : "train" | "val" | "test" — uses CFG split fractions
    trade_log  : path to saved trade-log CSV (validation by default)
    equity_csv : path to saved equity-curve CSV (validation by default)
    out_dir    : where to write the slice HTML files
    open_browser : auto-open the candle+trades chart in the default browser
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Features (fast reload) ────────────────────────────────────────────
    print("Loading features …")
    feat, feature_cols = _load_features()

    # ── 2. Resolve start / end from split name if given ─────────────────────
    if split is not None:
        t0, t1 = _split_bounds(feat, split)
        start = start or t0
        end   = end   or t1
        print(f"  Split '{split}':  {t0}  →  {t1}")

    # ── 3. Filter feature frame ──────────────────────────────────────────────
    feat_slice = _filter_feat(feat, start, end, bars, CFG.source_tz)
    if feat_slice.empty:
        raise ValueError("No bars in the requested slice — check your date range.")
    t_start = feat_slice.index.min()
    t_end   = feat_slice.index.max()
    print(f"  Slice: {t_start}  →  {t_end}  ({len(feat_slice):,} bars)")

    # ── 4. Load & filter saved simulation outputs ────────────────────────────
    trades = _load_trades(Path(trade_log))
    equity = _load_equity(Path(equity_csv))
    trades_slice = _filter_trades(trades, t_start, t_end)
    equity_slice = _filter_equity(equity, t_start, t_end)

    n_trades = len(trades_slice)
    print(f"  Trades in slice: {n_trades}")

    # ── 5. Performance summary for this slice ─────────────────────────────────
    if not equity_slice.empty:
        rep = full_report(equity_slice, trades_slice,
                          initial_equity=float(equity_slice["equity"].iloc[0]),
                          periods_per_year=CFG.periods_per_year)
        print("\n  Performance for this slice:")
        print(rep.to_string())

    # ── 6. Generate HTML charts ───────────────────────────────────────────────
    slug = _slug(start, end, split, bars)

    charts = {
        f"view_{slug}_01_candles_trades.html": plot_trades_on_chart(
            feat_slice, trades_slice,
            n=len(feat_slice),
            title=f"Trades on chart  [{slug}]",
        ),
        f"view_{slug}_02_candles_indicators.html": plot_candles_with_indicators(
            feat_slice,
            n=len(feat_slice),
            title=f"Candles + indicators  [{slug}]",
        ),
        f"view_{slug}_03_equity_drawdown.html": plot_equity_and_drawdown(
            equity_slice,
            title=f"Equity & drawdown  [{slug}]",
        ),
        f"view_{slug}_04_feature_correlation.html": plot_feature_correlation(
            feat_slice, feature_cols,
            title=f"Feature correlation  [{slug}]",
        ),
    }

    first_path = None
    for fname, fig in charts.items():
        p = save_html(fig, out_dir / fname)
        if first_path is None:
            first_path = p
        print(f"  Saved: {p}")

    if open_browser and first_path:
        webbrowser.open(first_path.resolve().as_uri())

    return out_dir


def _slug(
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    split: str | None,
    bars: int | None,
) -> str:
    if split:
        return split
    parts = []
    if start:
        parts.append(str(start)[:10])
    if end:
        parts.append(str(end)[:10])
    if bars and not (start or end):
        parts.append(f"last{bars}bars")
    return "_".join(parts) if parts else "slice"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise a time slice of saved simulation results."
    )
    p.add_argument("--start",      default=None, help="Start date e.g. 2024-01-01")
    p.add_argument("--end",        default=None, help="End date   e.g. 2024-06-01")
    p.add_argument("--bars",       default=None, type=int,
                   help="Show last N decision bars (after date filtering)")
    p.add_argument("--split",      default=None,
                   choices=["train", "val", "test"],
                   help="Show the train/val/test split as defined by CFG")
    p.add_argument("--trade-log",  default=str(_OUT / "selected_val_trade_log.csv"))
    p.add_argument("--equity-csv", default=str(_OUT / "selected_val_equity_curve.csv"))
    p.add_argument("--out-dir",    default=str(_VIEW))
    p.add_argument("--no-browser", action="store_true",
                   help="Save HTML files but do not open the browser")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    view_slice(
        start=args.start,
        end=args.end,
        bars=args.bars if args.bars else (500 if not (args.start or args.end or args.split) else None),
        split=args.split,
        trade_log=args.trade_log,
        equity_csv=args.equity_csv,
        out_dir=args.out_dir,
        open_browser=not args.no_browser,
    )
