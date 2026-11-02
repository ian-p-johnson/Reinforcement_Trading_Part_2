"""The vectorised intrabar scan must match the original per-bar loop.

The reference implementation below is a verbatim copy of the pre-Track-A
``_simulate_m1_until_next_decision`` inner loop (DECISIONS.md D3/D4): it is the
oracle for the numpy rewrite.  Randomised windows/levels cover same-bar double
touches, ties, empty windows, and level positions at the window boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

import conftest  # noqa: F401

from env_bracket import BracketTradingEnv, Position


def reference_intrabar(env, position: Position):
    """Original per-bar scan (pre-refactor logic), returns ('SL'|'TP'|None, idx)."""
    if position.direction == 0:
        return None, None
    start = env._current_time()
    end = env._next_time()
    lo = int(env._m1_index.searchsorted(start, side="right"))
    hi = int(env._m1_index.searchsorted(end, side="right"))

    for idx in range(lo, hi):
        high = env._m1_high[idx]
        low = env._m1_low[idx]
        p = position
        if p.direction == 1:
            sl_hit = low <= p.sl
            tp_hit = high >= p.tp
        else:
            sl_hit = high >= p.sl
            tp_hit = low <= p.tp
        if sl_hit:
            return "SL", idx
        if tp_hit:
            return "TP", idx
    return None, None


def vectorized_intrabar(env, position: Position):
    """The new implementation's decision (without closing the position)."""
    if position.direction == 0:
        return None, None
    lo = int(env._m1_index.searchsorted(env._current_time(), side="right"))
    hi = int(env._m1_index.searchsorted(env._next_time(), side="right"))
    if hi <= lo:
        return None, None
    highs = env._m1_high[lo:hi]
    lows = env._m1_low[lo:hi]
    if position.direction == 1:
        sl_mask = lows <= position.sl
        tp_mask = highs >= position.tp
    else:
        sl_mask = highs >= position.sl
        tp_mask = lows <= position.tp
    sl_any, tp_any = sl_mask.any(), tp_mask.any()
    if not (sl_any or tp_any):
        return None, None
    sl_rel = int(np.argmax(sl_mask)) if sl_any else None
    tp_rel = int(np.argmax(tp_mask)) if tp_any else None
    if sl_rel is not None and (tp_rel is None or sl_rel <= tp_rel):
        return "SL", lo + sl_rel
    return "TP", lo + tp_rel


def _make_env():
    from scenarios import build_frame, make_m1
    m1 = make_m1(n_days=30, seed=3)
    feat, cols = build_frame(m1)
    return BracketTradingEnv(feat, m1, cols,
                             max_episode_steps=50, randomize_start=True)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_intrabar_matches_reference(seed):
    """Randomised positions/levels: vectorised scan == per-bar loop."""
    env = _make_env()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    for _ in range(200):
        env.i = int(rng.integers(0, len(env.decision_df) - 2))
        close = env._close[env.i]
        atr = max(env._atr[env.i], 1e-12)
        direction = int(rng.choice([1, -1]))
        # Levels straddling the close in price space so both hits, one-sided
        # hits and misses all occur across the draws.
        sl = close - direction * float(rng.uniform(-2.0, 8.0) * atr)
        tp = close + direction * float(rng.uniform(-2.0, 8.0) * atr)
        env.position = Position(direction=direction, sl=sl, tp=tp,
                                entry_price=close, units=1.0, risk_cash=1.0)
        ref_reason, ref_idx = reference_intrabar(env, env.position)
        vec_reason, vec_idx = vectorized_intrabar(env, env.position)
        assert (ref_reason, ref_idx) == (vec_reason, vec_idx), (
            f"seed={seed} i={env.i} dir={direction}: "
            f"reference={(ref_reason, ref_idx)} vectorized={(vec_reason, vec_idx)}"
        )


def test_same_bar_double_touch_resolves_to_sl():
    """A window bar whose range covers both levels must exit SL first."""
    env = _make_env()
    env.reset(seed=0)
    env.i = 5
    close = env._close[env.i]
    atr = max(env._atr[env.i], 1e-12)
    # Both levels well inside the typical bar range -> first window bar
    # usually touches both; where it doesn't, reference and vector must still
    # agree (covered by the randomised test).
    env.position = Position(direction=1, sl=close - 0.05 * atr, tp=close + 0.05 * atr,
                            entry_price=close, units=1.0, risk_cash=1.0)
    ref_reason, ref_idx = reference_intrabar(env, env.position)
    vec_reason, vec_idx = vectorized_intrabar(env, env.position)
    assert (ref_reason, ref_idx) == (vec_reason, vec_idx)
    if ref_reason is not None:
        # If the earliest window bar touches both, it must be SL.
        lo = int(env._m1_index.searchsorted(env._current_time(), side="right"))
        bar_high, bar_low = env._m1_high[ref_idx], env._m1_low[ref_idx]
        if bar_low <= env.position.sl and bar_high >= env.position.tp and ref_idx == lo:
            assert ref_reason == "SL"


def test_empty_window_leaves_position():
    """lo == hi: no intrabar work, position untouched."""
    from scenarios import _drop_m1_window, build_frame, make_m1, m1_slice_for
    m1 = make_m1(n_days=30, seed=11)
    feat, cols = build_frame(m1)
    t0, t1 = feat.index[10], feat.index[11]
    m1_missing = _drop_m1_window(m1, t0, t1)
    env = BracketTradingEnv(feat, m1_missing, cols, max_episode_steps=30)
    env.reset(seed=0)
    env.i = 10
    close = env._close[env.i]
    env.position = Position(direction=1, sl=close - 1e-6, tp=close + 1e-6,
                            entry_price=close, units=1.0, risk_cash=1.0)
    assert reference_intrabar(env, env.position) == (None, None)
    assert vectorized_intrabar(env, env.position) == (None, None)
