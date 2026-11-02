"""Deterministic test scenarios for the bracket-trading environment.

These scenarios are the shared *inputs* for the golden-fixture generator
(``make_golden.py``) and the parity tests (``test_env_parity.py``).  Everything
here is synthetic and seeded: no external data is required.

Design notes (see DECISIONS.md):
  * D2  — synthetic data keeps tests hermetic and lets us build edge cases.
  * D9  — traces are captured per step (npz) plus a trade log (csv).
  * D10 — edge cases are scripted: immediate SL, same-candle SL+TP (SL-first
    rule), flip, manual close, truncation, termination at data end, a missing
    M1 window (empty intrabar window), a multi-day data gap, and a randomised
    episode start.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from data_loader import resample_ohlcv
from env_bracket import BracketTradingEnv
from features import prepare_feature_frame

SOURCE_TZ = "UTC"
WARMUP_BARS = 250


# ── synthetic M1 data ────────────────────────────────────────────────────────

def make_m1(n_days: int = 30, seed: int = 7, start: str = "2024-01-02",
            start_price: float = 2000.0, sigma: float = 0.6) -> pd.DataFrame:
    """Deterministic M1 OHLCV random walk with an open/close/high/low structure.

    ``sigma`` is the per-minute vol in price units; ATR ends up around
    ``sigma * sqrt(1440)``-ish which gives realistic SL/TP distances for
    spread=0.2 / slippage=0.02.
    """
    n = n_days * 1440
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq="1min", tz=SOURCE_TZ)
    close = start_price + np.cumsum(rng.normal(0.0, sigma, size=n))
    open_ = np.empty_like(close)
    open_[0] = start_price
    open_[1:] = close[:-1]
    noise = np.abs(rng.normal(0.0, sigma, size=n))
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    vol = rng.gamma(2.0, 0.5, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": vol},
        index=pd.DatetimeIndex(idx, name="Time"),
    )


def build_frame(m1: pd.DataFrame, warmup: int = WARMUP_BARS):
    """M1 -> H1 decision bars -> causal feature frame (pipeline-faithful)."""
    decision = resample_ohlcv(m1, "1h")
    feat, feature_cols = prepare_feature_frame(decision, warmup_bars=warmup)
    return feat, feature_cols


def m1_slice_for(m1: pd.DataFrame, decision_df: pd.DataFrame) -> pd.DataFrame:
    """Same windowing rule as train_ppo._slice_m1_for_decision_window."""
    if decision_df.empty:
        return m1.iloc[0:0].copy()
    start, end = decision_df.index.min(), decision_df.index.max()
    return m1.loc[(m1.index > start) & (m1.index <= end)].copy()


# ── scenario definition ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    m1: pd.DataFrame
    decision_df: pd.DataFrame
    feature_cols: List[str]
    actions: np.ndarray                 # (n, 3) int
    env_kwargs: Dict = field(default_factory=dict)
    reset_seed: Optional[int] = 0
    notes: str = ""

    def make_env(self) -> BracketTradingEnv:
        kw = dict(
            sl_atr_multipliers=(1.0, 1.5, 2.0),
            tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
            initial_equity=10_000.0,
            risk_fraction=0.005,
            spread_price=0.20,
            slippage_price=0.02,
            commission_per_trade=0.01,
            holding_penalty=0.00002,
            reward_mtm_weight=0.01,
            randomize_start=False,
            max_episode_steps=None,
        )
        kw.update(self.env_kwargs)
        return BracketTradingEnv(
            self.decision_df, m1_slice_for(self.m1, self.decision_df),
            self.feature_cols, **kw,
        )


# ── trace capture (shared by fixture writer and parity test) ─────────────────

TRACE_KEYS = ("obs", "reward", "terminated", "truncated", "equity",
              "realized_pnl", "direction")


def run_trace(scenario: Scenario) -> Dict[str, np.ndarray]:
    """Replay the scenario's actions and capture a full per-step trace."""
    env = scenario.make_env()
    obs, _ = env.reset(seed=scenario.reset_seed)
    start_i = int(env.i)

    obs_l, rew_l, term_l, trunc_l = [np.asarray(obs, dtype=np.float32)], [], [], []
    eq_l, real_l, dir_l = [], [], []

    for action in scenario.actions:
        obs, reward, terminated, truncated, _info = env.step(action)
        obs_l.append(np.asarray(obs, dtype=np.float32))
        rew_l.append(float(reward))
        term_l.append(bool(terminated))
        trunc_l.append(bool(truncated))
        eq_l.append(float(env.equity))
        real_l.append(float(env.realized_pnl))
        dir_l.append(int(env.position.direction))
        if terminated or truncated:
            break

    trace = {
        "obs": np.stack(obs_l),                       # includes reset obs at [0]
        "reward": np.asarray(rew_l, dtype=np.float64),
        "terminated": np.asarray(term_l, dtype=bool),
        "truncated": np.asarray(trunc_l, dtype=bool),
        "equity": np.asarray(eq_l, dtype=np.float64),
        "realized_pnl": np.asarray(real_l, dtype=np.float64),
        "direction": np.asarray(dir_l, dtype=np.int8),
        "start_i": np.asarray(start_i),
    }
    hist = env.equity_curve()
    trace["hist_equity"] = hist["equity"].to_numpy(dtype=np.float64) if len(hist) else np.empty(0)
    trace["hist_realized"] = hist["realized_pnl"].to_numpy(dtype=np.float64) if len(hist) else np.empty(0)
    trace["hist_close"] = hist["close"].to_numpy(dtype=np.float64) if len(hist) else np.empty(0)
    trace["hist_reward"] = hist["reward"].to_numpy(dtype=np.float64) if len(hist) else np.empty(0)
    trace["hist_position"] = hist["position"].to_numpy(dtype=np.int8) if len(hist) else np.empty(0, dtype=np.int8)
    trace["hist_time_ns"] = (hist.index.asi8 if len(hist)
                             else np.empty(0, dtype=np.int64))
    return trace, env.trade_log()


# ── edge-case data crafting ──────────────────────────────────────────────────

def _widen_first_bar_in_window(m1: pd.DataFrame, window_start: pd.Timestamp,
                               window_end: pd.Timestamp, half_width: float) -> pd.DataFrame:
    """Make the first M1 bar of (window_start, window_end] span +/- half_width.

    Used to force SL and TP to be touched inside the *same* M1 candle, which
    exercises the pessimistic SL-first rule.
    """
    m1 = m1.copy()
    lo = int(m1.index.searchsorted(window_start, side="right"))
    hi = int(m1.index.searchsorted(window_end, side="right"))
    assert hi > lo, "target window contains no M1 bars"
    i = lo
    mid = float(m1["Close"].iloc[i])
    m1.iloc[i, m1.columns.get_loc("High")] = mid + half_width
    m1.iloc[i, m1.columns.get_loc("Low")] = mid - half_width
    # keep the bar internally consistent
    o, c = float(m1["Open"].iloc[i]), float(m1["Close"].iloc[i])
    m1.iloc[i, m1.columns.get_loc("High")] = max(mid + half_width, o, c)
    m1.iloc[i, m1.columns.get_loc("Low")] = min(mid - half_width, o, c)
    return m1


def _drop_m1_window(m1: pd.DataFrame, window_start: pd.Timestamp,
                    window_end: pd.Timestamp) -> pd.DataFrame:
    """Remove every M1 bar inside (window_start, window_end] (empty window)."""
    m1 = m1.copy()
    lo = int(m1.index.searchsorted(window_start, side="right"))
    hi = int(m1.index.searchsorted(window_end, side="right"))
    return m1.drop(m1.index[lo:hi])


def _rng_actions(n: int, seed: int, sl_n: int = 3, tp_n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.integers(0, 3, size=n),
        rng.integers(0, sl_n, size=n),
        rng.integers(0, tp_n, size=n),
    ]).astype(int)


# ── scenario catalogue ───────────────────────────────────────────────────────

def build_scenarios() -> List[Scenario]:
    scenarios: List[Scenario] = []

    # Shared, uncrafted dataset for most scenarios.
    base_m1 = make_m1(n_days=30, seed=7)
    base_feat, base_cols = build_frame(base_m1)

    # 1) flat: no exposure at all — exercises the no-position paths.
    scenarios.append(Scenario(
        name="flat",
        m1=base_m1, decision_df=base_feat, feature_cols=base_cols,
        actions=np.tile(np.array([[0, 1, 1]]), (600, 1)),
        notes="all-flat actions; zero trades expected",
    ))

    # 2) random: seeded mix of directions/brackets incl. holds and flips.
    scenarios.append(Scenario(
        name="random_seed7",
        m1=base_m1, decision_df=base_feat, feature_cols=base_cols,
        actions=_rng_actions(600, seed=7),
        notes="seeded random actions over the full action space",
    ))

    # 3) long tight-bracket: re-enters after every SL/TP exit.
    scenarios.append(Scenario(
        name="long_tight_reentry",
        m1=base_m1, decision_df=base_feat, feature_cols=base_cols,
        actions=np.tile(np.array([[1, 0, 0]]), (400, 1)),
        env_kwargs=dict(max_episode_steps=400),
        notes="constant long; SL=1.0*ATR, TP=1R -> churn of exits and re-entries",
    ))

    # 4) same-candle SL+TP: crafted wide first bar in the entry window.
    #    Build the dataset, locate the step-0 window from the *env's* decision
    #    frame, widen its first M1 bar, then rebuild features on the crafted M1.
    wide_m1 = base_m1.copy()
    probe_feat, probe_cols = build_frame(base_m1)
    t0, t1 = probe_feat.index[0], probe_feat.index[1]
    entry_close = float(probe_feat["Close"].iloc[0])
    entry_atr = float(probe_feat["atr"].iloc[0])
    wide_m1 = _widen_first_bar_in_window(wide_m1, t0, t1,
                                         half_width=50.0 * entry_atr)
    wide_feat, wide_cols = build_frame(wide_m1)
    scenarios.append(Scenario(
        name="same_candle_sl_tp",
        m1=wide_m1, decision_df=wide_feat, feature_cols=wide_cols,
        actions=np.tile(np.array([[1, 0, 0]]), (30, 1)),
        notes=(f"entry close={entry_close:.3f} atr={entry_atr:.3f}; first M1 "
               "bar of the entry window spans +/-50*ATR so SL and TP are "
               "touched in the same candle -> exit reason must be SL"),
    ))

    # 5) flip + manual close + re-entry with wide brackets.
    flip_actions = np.concatenate([
        np.tile(np.array([[1, 1, 1]]), (5, 1)),      # open long
        np.tile(np.array([[2, 1, 1]]), (100, 1)),    # flip to short, hold
        np.tile(np.array([[0, 1, 1]]), (20, 1)),     # manual close, stay flat
        np.tile(np.array([[1, 2, 2]]), (25, 1)),     # re-enter long, wide
    ])
    scenarios.append(Scenario(
        name="flip_and_close",
        m1=base_m1, decision_df=base_feat, feature_cols=base_cols,
        actions=flip_actions,
        notes="long -> flip short -> manual close -> re-enter long",
    ))

    # 6) truncation boundary at 40 steps.
    scenarios.append(Scenario(
        name="truncation_40",
        m1=base_m1, decision_df=base_feat, feature_cols=base_cols,
        actions=np.tile(np.array([[1, 1, 1]]), (200, 1)),
        env_kwargs=dict(max_episode_steps=40),
        reset_seed=11,
        notes="episode truncates at step 40 (mid-trade) -> truncated=True",
    ))

    # 7) termination at the data end (short decision tail).
    tail_feat = base_feat.iloc[-25:]
    tail_m1 = base_m1
    scenarios.append(Scenario(
        name="data_end",
        m1=tail_m1, decision_df=tail_feat, feature_cols=base_cols,
        actions=np.tile(np.array([[1, 0, 3]]), (50, 1)),
        notes="25-bar decision tail; episode must terminate at the data end",
    ))

    # 8) missing M1 window: the intrabar window for one decision step is empty.
    #    Keep the decision frame from the full series but drop the M1 bars of
    #    the window at decision index 150 -> lo == hi -> no TP/SL check there.
    miss_m1 = _drop_m1_window(base_m1, base_feat.index[150], base_feat.index[151])
    scenarios.append(Scenario(
        name="missing_m1_window",
        m1=miss_m1, decision_df=base_feat, feature_cols=base_cols,
        actions=np.tile(np.array([[1, 1, 3]]), (300, 1)),
        notes="M1 bars of the window at decision index 150 removed; position "
              "must survive the empty window",
    ))

    # 9) multi-day gap (weekend-style hole) crossed while holding.
    gap_m1 = base_m1.copy()
    gap_start = base_feat.index[100]
    gap_m1 = gap_m1.drop(gap_m1.loc[gap_start:gap_start + pd.Timedelta(days=3)].index)
    gap_feat, gap_cols = build_frame(gap_m1)
    scenarios.append(Scenario(
        name="gap_3days",
        m1=gap_m1, decision_df=gap_feat, feature_cols=gap_cols,
        actions=np.tile(np.array([[1, 1, 3]]), (600, 1)),
        notes="3-day hole in the M1 feed while holding long",
    ))

    # 10) randomised episode start (randomize_start=True needs a shorter
    #     max_episode_steps to actually randomise — see env reset headroom).
    scenarios.append(Scenario(
        name="random_start",
        m1=base_m1, decision_df=base_feat, feature_cols=base_cols,
        actions=_rng_actions(300, seed=5),
        env_kwargs=dict(randomize_start=True, max_episode_steps=300),
        reset_seed=123,
        notes="randomize_start=True with fixed reset seed -> deterministic start",
    ))

    return scenarios


SCENARIOS: List[Scenario] = build_scenarios()
