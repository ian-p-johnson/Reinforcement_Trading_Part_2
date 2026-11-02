"""Data-cache invariants (Track A5, DECISIONS.md D16).

The cache must return byte-identical data to the fresh computation and must
invalidate when the source file or the relevant config changes.
"""
from __future__ import annotations

import pytest

import conftest  # noqa: F401

import data_cache as dc
from config import CFG
from data_loader import resample_ohlcv
from features import prepare_feature_frame


@pytest.fixture(scope="module")
def cached():
    return dc.load_all()


def test_cache_matches_fresh_computation(cached):
    m1_c, _dec_c, feat_c, cols_c = cached
    if not CFG.csv_path.exists():
        pytest.skip("real CSV not present")
    m1_f = dc.load_mt_ohlcv_csv(
        CFG.csv_path, time_col=CFG.time_col, source_tz=CFG.source_tz,
        timestamp_is_bar_open=CFG.timestamp_is_bar_open,
        bar_duration=CFG.pandas_execution_tf,
        start_date=CFG.start_date, end_date=CFG.end_date,
        max_days_for_demo=CFG.max_days_for_demo,
    )
    assert m1_c.index.equals(m1_f.index), "cached M1 index differs from fresh"
    assert (m1_c.to_numpy() == m1_f.to_numpy()).all(), "cached M1 values differ from fresh"

    dec_f = resample_ohlcv(m1_f, CFG.pandas_tf)
    feat_f, cols_f = prepare_feature_frame(dec_f, CFG.warmup_bars,
                                           CFG.atr_period, CFG.rsi_period)
    assert cols_c == list(cols_f), "cached feature columns differ"
    assert feat_c.index.equals(feat_f.index)
    assert (feat_c[cols_c].to_numpy() == feat_f[cols_f].to_numpy()).all(), \
        "cached feature values differ from fresh"


def test_cache_keys_change_with_config():
    k1 = dc.m1_key()
    old = CFG.max_days_for_demo
    try:
        CFG.max_days_for_demo = 123
        k2 = dc.m1_key()
    finally:
        CFG.max_days_for_demo = old
    assert k1 != k2, "m1_key must respond to parse-parameter changes"

    mk = dc.m1_key()
    f1 = dc.feature_key(mk)
    old_warm = CFG.warmup_bars
    try:
        CFG.warmup_bars = old_warm + 7
        f2 = dc.feature_key(mk)
    finally:
        CFG.warmup_bars = old_warm
    assert f1 != f2, "feature_key must respond to feature-parameter changes"


def test_cache_roundtrip_index_preserved():
    df = dc.load_m1_cached()
    assert df.index.tz is not None, "tz-aware index lost in parquet round-trip"
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
