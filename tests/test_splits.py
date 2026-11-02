"""Split/embargo invariants — the walk-forward scaffolding must stay sound.

Covers the claims the rest of the project relies on (ANALYSIS.md §7):
  * the sealed test is byte-identical between the single-split and block
    walk-forward schemes (``test_frac == 1 - train_frac - val_frac``),
  * embargo gaps keep train/val/test disjoint,
  * sliding folds are chronological with contiguous test windows when
    ``step == test`` (the stitched OOS curve property).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import conftest  # noqa: F401

from data_loader import (
    make_sliding_folds,
    make_walk_forward_folds,
    split_train_val_test,
)
from features import prepare_feature_frame
from leakage_checks import assert_feature_stability_when_future_appended
from scenarios import build_frame, make_m1

FRACS = dict(train_frac=0.8, val_frac=0.1, test_frac=0.1)


@pytest.fixture(scope="module")
def feat_frame():
    m1 = make_m1(n_days=400, seed=42)
    feat, cols = build_frame(m1)
    return feat, cols


def test_sealed_test_identical_across_schemes(feat_frame):
    feat, _ = feat_frame
    train, val, test = split_train_val_test(
        feat, train_frac=FRACS["train_frac"], val_frac=FRACS["val_frac"],
        embargo_bars=200)
    folds, wf_test = make_walk_forward_folds(
        feat, n_folds=5, test_frac=FRACS["test_frac"],
        embargo_bars=200, anchored=False)
    assert test.index.equals(wf_test.index), \
        "sealed test must be byte-identical between the two schemes"
    assert len(folds) == 5


def test_split_disjoint_with_embargo(feat_frame):
    feat, _ = feat_frame
    embargo = 200
    train, val, test = split_train_val_test(
        feat, train_frac=0.8, val_frac=0.1, embargo_bars=embargo)
    # Disjointness
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    # Embargo size: the raw cut points are 80% and 90% of the frame.
    n = len(feat)
    raw_train_end = feat.index[int(n * 0.8) - 1]
    raw_val_start = feat.index[int(n * 0.8)]
    gap_bars = feat.index.get_loc(raw_val_start) - feat.index.get_loc(raw_train_end) - 1
    assert train.index.max() <= raw_train_end
    assert val.index.min() >= raw_val_start
    # At least `embargo` bars were dropped on each side of the boundary.
    assert feat.index.get_loc(val.index.min()) - feat.index.get_loc(train.index.max()) - 1 >= 2 * embargo - 1


def test_walk_forward_folds_ordered_and_disjoint(feat_frame):
    feat, _ = feat_frame
    folds, test = make_walk_forward_folds(feat, n_folds=5, test_frac=0.1,
                                          embargo_bars=200, anchored=False)
    prev_val_end = None
    for k, (tr, va) in enumerate(folds, start=1):
        assert len(tr) and len(va), f"fold {k} empty"
        assert tr.index.max() < va.index.min(), f"fold {k} train/val overlap"
        if prev_val_end is not None:
            assert prev_val_end < va.index.min(), "validation windows must roll forward"
        prev_val_end = va.index.max()
        assert va.index.max() < test.index.min(), f"fold {k} touches sealed test"


def test_sliding_folds_chronological_with_contiguous_tests(feat_frame):
    feat, _ = feat_frame
    embargo = 200
    folds = make_sliding_folds(feat, train_years=0.5, val_months=1,
                               test_months=1, step_months=1, embargo_bars=embargo)
    assert len(folds) >= 3
    prev_test_end = None
    for k, (tr, va, te) in enumerate(folds, start=1):
        assert tr.index.max() < va.index.min(), f"fold {k} train/val overlap"
        assert va.index.max() < te.index.min(), f"fold {k} val/test overlap"
        assert te.index.min() < te.index.max()
        if prev_test_end is not None:
            # step == test -> nominal test windows are contiguous; the embargo
            # purges the first `embargo_bars` of each window, so consecutive
            # test slices are separated by exactly the embargo in bar terms.
            gap = (feat.index.get_loc(te.index.min())
                   - feat.index.get_loc(prev_test_end) - 1)
            assert gap == embargo, \
                f"fold {k}: expected {embargo}-bar embargo gap, got {gap}"
        prev_test_end = te.index.max()


def test_features_are_causal():
    """Appending future bars must not change past features (leakage guard)."""
    m1 = make_m1(n_days=30, seed=5)
    from data_loader import resample_ohlcv
    decision = resample_ohlcv(m1, "1h")
    from features import add_stationary_features
    _probe, feature_cols = add_stationary_features(decision)
    cut = 300
    assert assert_feature_stability_when_future_appended(
        decision.iloc[:cut + 80], feature_cols=feature_cols, cut_index=cut,
        atr_period=14, rsi_period=14)
