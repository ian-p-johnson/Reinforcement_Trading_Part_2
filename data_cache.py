"""Parquet-backed data cache (Track A5, DECISIONS.md D16).

Every entry point in the project re-parsed the 500 MB M1 CSV on each run
(ANALYSIS.md §8: ~3.5-6.6 s per load, x5 entry points per workflow).  This
module caches the parsed M1 frame and the derived H1 feature frame in
``data/cache/*.parquet`` keyed by a hash of the source file (size + mtime) and
the parsing/feature parameters, so a changed CSV or config invalidates
automatically.

Usage::

    from data_cache import load_all
    m1, decision, feat, feature_cols = load_all()

CLI::

    python data_cache.py            # show cache status
    python data_cache.py --rebuild  # recompute and overwrite
    python data_cache.py --clear    # delete cached files
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import CFG
from data_loader import load_mt_ohlcv_csv, resample_ohlcv
from features import prepare_feature_frame

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"


# ── cache keys ───────────────────────────────────────────────────────────────

def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def m1_key() -> str:
    """Key for the parsed M1 frame: source file + parse parameters."""
    csv = Path(CFG.csv_path)
    return _hash({
        "fingerprint": _file_fingerprint(csv) if csv.exists() else "synthetic",
        "time_col": CFG.time_col,
        "source_tz": CFG.source_tz,
        "timestamp_is_bar_open": CFG.timestamp_is_bar_open,
        "bar_duration": CFG.pandas_execution_tf,
        "start_date": CFG.start_date,
        "end_date": CFG.end_date,
        "max_days_for_demo": CFG.max_days_for_demo,
    })


def feature_key(mkey: str) -> str:
    """Key for the H1 feature frame: M1 key + resampling + feature params."""
    return _hash({
        "m1": mkey,
        "decision_timeframe": CFG.pandas_tf,
        "warmup_bars": CFG.warmup_bars,
        "atr_period": CFG.atr_period,
        "rsi_period": CFG.rsi_period,
    })


# ── load / store ─────────────────────────────────────────────────────────────

def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(path)   # atomic-ish: never leave a torn cache file


def load_m1_cached(force: bool = False) -> pd.DataFrame:
    key = m1_key()
    path = CACHE_DIR / f"m1_{key}.parquet"
    if not force and path.exists():
        return _read_parquet(path)
    m1 = load_mt_ohlcv_csv(
        CFG.csv_path,
        time_col=CFG.time_col,
        source_tz=CFG.source_tz,
        timestamp_is_bar_open=CFG.timestamp_is_bar_open,
        bar_duration=CFG.pandas_execution_tf,
        start_date=CFG.start_date,
        end_date=CFG.end_date,
        max_days_for_demo=CFG.max_days_for_demo,
    )
    _write_parquet(m1, path)
    return m1


def load_feature_frame_cached(
    force: bool = False,
) -> Tuple[pd.DataFrame, list]:
    """Cached (feature_frame, feature_cols) for the current config."""
    mkey = m1_key()
    fkey = feature_key(mkey)
    fpath = CACHE_DIR / f"feat_{fkey}.parquet"
    mpath = CACHE_DIR / f"feat_{fkey}_cols.json"
    if not force and fpath.exists() and mpath.exists():
        feat = _read_parquet(fpath)
        cols = json.loads(mpath.read_text())
        return feat, cols

    m1 = load_m1_cached(force=force)
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, cols = prepare_feature_frame(
        decision,
        warmup_bars=CFG.warmup_bars,
        atr_period=CFG.atr_period,
        rsi_period=CFG.rsi_period,
    )
    _write_parquet(feat, fpath)
    mpath.write_text(json.dumps(list(cols)))
    return feat, list(cols)


def load_all(
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """(m1, decision, feat, feature_cols) — the full pipeline data triple.

    ``decision`` is resampled from M1 on the fly (cheap, ~0.3 s) so it is not
    cached separately; M1 and the feature frame are the expensive parts.
    """
    m1 = load_m1_cached(force=force)
    feat, cols = load_feature_frame_cached(force=force)
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    return m1, decision, feat, cols


def cache_files() -> list:
    return sorted(CACHE_DIR.glob("*.parquet")) + sorted(CACHE_DIR.glob("*_cols.json"))


def clear() -> int:
    n = 0
    for p in cache_files():
        p.unlink()
        n += 1
    return n


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    if args.clear:
        print(f"removed {clear()} cache file(s)")
        return 0

    if args.rebuild:
        clear()
    m1, decision, feat, cols = load_all(force=args.rebuild)
    total = sum(p.stat().st_size for p in cache_files()) / 1e6
    print(f"cache dir : {CACHE_DIR}")
    print(f"files     : {len(cache_files())} ({total:.1f} MB)")
    print(f"M1        : {len(m1):,} rows  {m1.index.min()} -> {m1.index.max()}")
    print(f"decision  : {len(decision):,} bars ({CFG.pandas_tf})")
    print(f"features  : {len(feat):,} bars x {len(cols)} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
