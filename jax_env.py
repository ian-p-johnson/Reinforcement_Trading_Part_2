"""JAX port of ``env_bracket.BracketTradingEnv`` (Track B2, SCALING_PLAN.md B2).

Functional gymnax-style API (DECISIONS.md D22):
    reset(rng, params)                -> (obs, state)      randomised start (training)
    reset_at(params, start_i)         -> (obs, state)      explicit start (eval / parity)
    step(state, action, params)       -> (obs, state, reward, terminated, truncated, info)
    step_autoreset(rng, state, action, params) -> same, with auto-reset + final_observation

Precision (D23): the simulation runs in float64 (``jax_enable_x64`` is enabled at import);
observations are assembled exactly like ``env_bracket._observation`` — sanitized float32
market features plus position features computed in f64 then cast to f32.

Intrabar TP/SL (D25/D29/D30): per-window padded gathers (LUT_SPEC.md §3.4), built on the
host with the same half-open ``searchsorted(side="right")`` window rule as
``env_bracket._simulate_m1_until_next_decision``.  Padding is ±inf so it can never register
a touch; same-bar double touches resolve to SL (pessimistic rule); empty windows hold.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

# Trade exit-reason ids (stable; used by the trade-log reconstructor).
REASON_NONE = 0
REASON_SL = 1
REASON_TP = 2
REASON_MANUAL = 3
REASON_FLIP = 4

# Sanity bound on the padded window width (D30): H1 decisions over M1 execution cap
# windows at 60 bars unless decision rows are dropped mid-series.  Loud failure >1024.
MAX_WINDOW_WIDTH = 1024


class EnvParams(NamedTuple):
    """Static per-dataset inputs (D25): replicated, never mutated during a rollout."""

    feat: Array            # f32 [T, n_features] — sanitized like env_bracket (D1/D2)
    close: Array           # f64 [T]
    atr: Array             # f64 [T]
    decision_ts: Array     # i64 [T]  decision-bar close times, ns
    high_pad: Array        # f64 [T, W]  M1 highs per window, padding -inf
    low_pad: Array         # f64 [T, W]  M1 lows per window, padding +inf
    ts_pad: Array          # i64 [T, W]  M1 close times per window, padding 0
    win_len: Array         # i32 [T]  real bars in each window
    sl_mults: Array        # f64 [n_sl]
    tp_mults: Array        # f64 [n_tp]
    initial_equity: Array  # f64 scalar
    risk_fraction: Array   # f64 scalar
    spread: Array          # f64 scalar (spread_price)
    slippage: Array        # f64 scalar
    commission: Array      # f64 scalar
    holding_penalty: Array  # f64 scalar
    mtm_weight: Array      # f64 scalar (reward_mtm_weight)
    max_episode_steps: Array  # i32 scalar


class EnvState(NamedTuple):
    """Per-env state as batched scalars (D24)."""

    i: Array               # i32 decision index
    steps: Array           # i32 steps since episode start
    equity: Array          # f64
    realized: Array        # f64 realized pnl
    direction: Array       # i32 +1 long / -1 short / 0 flat
    entry_ts: Array        # i64 ns (decision bar close time at entry)
    entry_price: Array     # f64
    sl: Array              # f64
    tp: Array              # f64
    units: Array           # f64
    risk_cash: Array       # f64
    tp_r: Array            # f64 planned TP R-multiple (bracket choice)
    sl_atr_mult: Array     # f64 planned SL ATR multiplier (bracket choice)
    bars_in_trade: Array   # i32


class StepInfo(NamedTuple):
    """Fixed-shape per-step info (D31): up to two trade records per step, in
    pandas-append order — slot 0 is the decision-close record (manual/flip),
    slot 1 the intrabar TP/SL record.  A flip followed by an intrabar exit on
    the same bar closes *two* positions in one step, exactly as ``env_bracket``
    appends two trades.  Scalars: post-step account state."""

    equity: Array          # f64 post-step
    realized: Array        # f64 post-step
    direction: Array       # i32 post-step
    closed: Array          # bool [2]
    reason: Array          # i32 [2] REASON_*
    trade_direction: Array  # i32 [2] +1 long / -1 short
    entry_ts: Array        # i64 [2]
    exit_ts: Array         # i64 [2]
    entry_price: Array     # f64 [2]
    exit_price: Array      # f64 [2] (post spread/slippage)
    sl: Array              # f64 [2]
    tp: Array              # f64 [2]
    sl_atr_mult: Array     # f64 [2]
    tp_r: Array            # f64 [2]
    units: Array           # f64 [2]
    pnl: Array             # f64 [2]
    r_mult: Array          # f64 [2]
    bars_in_trade: Array   # i32 [2] at close time


# ── host-side construction ──────────────────────────────────────────────────


def build_env_params(
    decision_df: pd.DataFrame,
    m1_df: pd.DataFrame,
    feature_cols: list[str],
    sl_atr_multipliers=(1.0, 1.5, 2.0),
    tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
    initial_equity: float = 10_000.0,
    risk_fraction: float = 0.005,
    spread_price: float = 0.20,
    slippage_price: float = 0.02,
    commission_per_trade: float = 0.0,
    holding_penalty: float = 0.00002,
    reward_mtm_weight: float = 0.01,
    max_episode_steps: int | None = None,
) -> EnvParams:
    """Build ``EnvParams`` from the same inputs as ``BracketTradingEnv.__init__``.

    Replicates the env's preprocessing bit-for-bit (D1/D2): dropna on features+atr,
    f32 feature matrix sanitized with nan=0/posinf=10/neginf=-10, f64 close/atr.
    Window rows bake the (decision[i], decision[i+1]] half-open M1 slice (D29).
    """
    decision_df = decision_df.dropna(subset=list(feature_cols) + ["atr"])
    feat32 = decision_df[list(feature_cols)].to_numpy(dtype=np.float32)
    feat32 = np.nan_to_num(feat32, nan=0.0, posinf=10.0, neginf=-10.0)
    close = np.ascontiguousarray(decision_df["Close"].to_numpy(dtype=np.float64))
    atr = np.ascontiguousarray(decision_df["atr"].to_numpy(dtype=np.float64))
    decision_ts = decision_df.index.asi8.astype(np.int64)

    m1_ts = np.ascontiguousarray(m1_df.index.asi8.astype(np.int64))
    m1_high = np.ascontiguousarray(m1_df["High"].to_numpy(dtype=np.float64))
    m1_low = np.ascontiguousarray(m1_df["Low"].to_numpy(dtype=np.float64))

    t = len(decision_df)
    if t < 3:
        raise ValueError(f"decision_df too small after dropna: {t} rows")

    end_ts = decision_ts[np.minimum(np.arange(t) + 1, t - 1)]
    lo = np.searchsorted(m1_ts, decision_ts, side="right")
    hi = np.searchsorted(m1_ts, end_ts, side="right")
    win_len = np.maximum(hi - lo, 0).astype(np.int32)

    w = int(win_len.max()) if t else 0
    if w > MAX_WINDOW_WIDTH:
        raise ValueError(
            f"window width {w} exceeds MAX_WINDOW_WIDTH={MAX_WINDOW_WIDTH}: "
            "decision rows appear to be missing mid-series — revisit D30"
        )

    high_pad = np.full((t, w), -np.inf, dtype=np.float64)
    low_pad = np.full((t, w), np.inf, dtype=np.float64)
    ts_pad = np.zeros((t, w), dtype=np.int64)
    for i in range(t):
        n = int(win_len[i])
        if n > 0:
            s, e = int(lo[i]), int(lo[i]) + n
            high_pad[i, :n] = m1_high[s:e]
            low_pad[i, :n] = m1_low[s:e]
            ts_pad[i, :n] = m1_ts[s:e]

    if max_episode_steps is None:
        max_episode_steps = t - 2

    return EnvParams(
        feat=jnp.asarray(feat32),
        close=jnp.asarray(close),
        atr=jnp.asarray(atr),
        decision_ts=jnp.asarray(decision_ts),
        high_pad=jnp.asarray(high_pad),
        low_pad=jnp.asarray(low_pad),
        ts_pad=jnp.asarray(ts_pad),
        win_len=jnp.asarray(win_len),
        sl_mults=jnp.asarray(np.asarray(sl_atr_multipliers, dtype=np.float64)),
        tp_mults=jnp.asarray(np.asarray(tp_r_multipliers, dtype=np.float64)),
        initial_equity=jnp.asarray(float(initial_equity), dtype=jnp.float64),
        risk_fraction=jnp.asarray(float(risk_fraction), dtype=jnp.float64),
        spread=jnp.asarray(float(spread_price), dtype=jnp.float64),
        slippage=jnp.asarray(float(slippage_price), dtype=jnp.float64),
        commission=jnp.asarray(float(commission_per_trade), dtype=jnp.float64),
        holding_penalty=jnp.asarray(float(holding_penalty), dtype=jnp.float64),
        mtm_weight=jnp.asarray(float(reward_mtm_weight), dtype=jnp.float64),
        max_episode_steps=jnp.asarray(int(max_episode_steps), dtype=jnp.int32),
    )


# ── functional API ──────────────────────────────────────────────────────────


def init_state(params: EnvParams, start_i: Array | int) -> EnvState:
    """Flat-position state at a given decision index (D22/D24)."""
    i = jnp.asarray(start_i, dtype=jnp.int32)
    z = jnp.asarray(0.0, dtype=jnp.float64)
    return EnvState(
        i=i,
        steps=jnp.asarray(0, dtype=jnp.int32),
        equity=params.initial_equity,
        realized=z,
        direction=jnp.asarray(0, dtype=jnp.int32),
        entry_ts=jnp.asarray(0, dtype=jnp.int64),
        entry_price=z,
        sl=z,
        tp=z,
        units=z,
        risk_cash=z,
        tp_r=z,
        sl_atr_mult=z,
        bars_in_trade=jnp.asarray(0, dtype=jnp.int32),
    )


def reset(rng: Array, params: EnvParams) -> tuple[Array, EnvState]:
    """Randomised-start reset: identical headroom rule to ``BracketTradingEnv.reset``."""
    t = params.close.shape[0]
    headroom = jnp.maximum(params.max_episode_steps + 2, 2)
    max_start = jnp.maximum(t - headroom, 1)
    start = jax.random.randint(rng, (), 0, max_start, dtype=jnp.int32)
    state = init_state(params, start)
    return observation(state, params), state


def observation(state: EnvState, params: EnvParams) -> Array:
    """float32 obs [n_features + 6], same assembly and sanitisation as env_bracket."""
    market = params.feat[state.i]
    close = params.close[state.i]
    atr = jnp.maximum(params.atr[state.i], 1e-12)
    d = state.direction

    def _pos_features():
        unrealized = (close - state.entry_price) * state.units * d.astype(jnp.float64)
        unrealized_r = unrealized / jnp.maximum(state.risk_cash, 1e-12)
        dist_tp_atr = ((state.tp - close) * d.astype(jnp.float64)) / atr
        dist_sl_atr = ((close - state.sl) * d.astype(jnp.float64)) / atr
        bars = state.bars_in_trade.astype(jnp.float64) / 100.0
        return jnp.stack([
            d.astype(jnp.float64),
            unrealized_r,
            jnp.minimum(bars, 10.0),
            dist_tp_atr,
            dist_sl_atr,
            state.tp_r,
        ])

    pos = _pos_features().astype(jnp.float32)
    pos = jnp.where(d == 0, jnp.zeros_like(pos), pos)
    obs = jnp.concatenate([market, pos]).astype(jnp.float32)
    return jnp.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)


def step(state: EnvState, action: Array, params: EnvParams):
    """One decision step — verbatim semantics of ``BracketTradingEnv.step``.

    (a) explicit close / flip at the decision close; (b) fresh entry; (c) intrabar
    TP/SL via the padded window gather (SL-first tie rule); (d) mark-to-market
    shaping reward; (e) position-state observation; (f) index increment with
    terminated (data end) / truncated (max_episode_steps) flags.
    """
    dir_raw = action[0].astype(jnp.int32)
    sl_idx = action[1].astype(jnp.int32)
    tp_idx = action[2].astype(jnp.int32)
    desired = jnp.where(dir_raw == 0, 0, jnp.where(dir_raw == 1, 1, -1)).astype(jnp.int32)

    prev_equity = state.equity
    reward_risk_unit = jnp.maximum(prev_equity * params.risk_fraction, 1e-12)
    close_i = params.close[state.i]
    ts_i = params.decision_ts[state.i]
    half = params.spread / 2.0 + params.slippage

    pos_dir = state.direction
    close_now = (pos_dir != 0) & (desired != pos_dir)
    open_now = (desired != 0) & (pos_dir != desired)

    # ── (a) close at the decision close (manual_close / flip_close) ─────────
    exit_price_d = close_i - pos_dir.astype(jnp.float64) * half
    pnl_d = (exit_price_d - state.entry_price) * state.units * pos_dir.astype(jnp.float64) - params.commission
    r_mult_d = pnl_d / jnp.maximum(state.risk_cash, 1e-12)
    equity_d = state.equity + pnl_d
    realized_d = state.realized + pnl_d
    reason_d = jnp.where(desired == 0, REASON_MANUAL, REASON_FLIP)

    # ── (b) open (uses equity after any close this step) ────────────────────
    equity_for_open = jnp.where(close_now, equity_d, state.equity)
    atr_i = jnp.maximum(params.atr[state.i], 1e-12)
    sl_atr_mult = params.sl_mults[sl_idx]
    sl_dist = jnp.maximum(sl_atr_mult * atr_i, 1e-8)
    tp_r = params.tp_mults[tp_idx]
    d_open = desired.astype(jnp.float64)
    entry_o = close_i + d_open * half
    sl_o = entry_o - d_open * sl_dist
    tp_o = entry_o + d_open * tp_r * sl_dist
    risk_o = jnp.maximum(equity_for_open * params.risk_fraction, 1e-8)
    units_o = risk_o / sl_dist

    sel_close = close_now
    sel_open = open_now

    equity1 = jnp.where(sel_open, equity_for_open, jnp.where(sel_close, equity_d, state.equity))
    realized1 = jnp.where(sel_close, realized_d, state.realized)
    direction1 = jnp.where(sel_open, desired, jnp.where(sel_close, jnp.asarray(0, dtype=jnp.int32), pos_dir))
    entry_ts1 = jnp.where(sel_open, ts_i, state.entry_ts)
    entry_price1 = jnp.where(sel_open, entry_o, state.entry_price)
    sl1 = jnp.where(sel_open, sl_o, state.sl)
    tp1 = jnp.where(sel_open, tp_o, state.tp)
    units1 = jnp.where(sel_open, units_o, state.units)
    risk1 = jnp.where(sel_open, risk_o, state.risk_cash)
    tp_r1 = jnp.where(sel_open, tp_r, state.tp_r)
    sl_atr_mult1 = jnp.where(sel_open, sl_atr_mult, state.sl_atr_mult)
    bars1 = jnp.where(sel_open, jnp.asarray(0, dtype=jnp.int32), state.bars_in_trade)

    # ── (c) intrabar first-touch over the window (state.i) ─────────────────
    highs = params.high_pad[state.i]
    lows = params.low_pad[state.i]
    ts_row = params.ts_pad[state.i]

    is_long = direction1 == 1
    sl_touch = jnp.where(is_long, lows <= sl1, highs >= sl1) & (direction1 != 0)
    tp_touch = jnp.where(is_long, highs >= tp1, lows <= tp1) & (direction1 != 0)

    sl_any = jnp.any(sl_touch)
    tp_any = jnp.any(tp_touch)
    sl_rel = jnp.argmax(sl_touch.astype(jnp.int32))
    tp_rel = jnp.argmax(tp_touch.astype(jnp.int32))
    sl_wins = sl_any & (~tp_any | (sl_rel <= tp_rel))
    touch = sl_any | tp_any
    close_at_sl = touch & sl_wins
    close_at_tp = touch & ~sl_wins
    hit = close_at_sl | close_at_tp

    rel = jnp.where(close_at_sl, sl_rel, tp_rel)
    exit_ts = ts_row[rel]
    exit_price_raw = jnp.where(close_at_sl, sl1, tp1)
    exit_price = exit_price_raw - direction1.astype(jnp.float64) * half
    pnl_m = (exit_price - entry_price1) * units1 * direction1.astype(jnp.float64) - params.commission
    r_mult_m = pnl_m / jnp.maximum(risk1, 1e-12)
    reason_m = jnp.where(close_at_sl, REASON_SL, REASON_TP)

    equity2 = equity1 + jnp.where(hit, pnl_m, 0.0)
    realized2 = realized1 + jnp.where(hit, pnl_m, 0.0)
    direction2 = jnp.where(hit, jnp.asarray(0, dtype=jnp.int32), direction1)
    bars2 = jnp.where(hit, jnp.asarray(0, dtype=jnp.int32), bars1)

    # Trade records (D31): slot 0 = decision-close (old position fields),
    # slot 1 = intrabar exit (position-after-open fields).
    rec_closed = jnp.stack([sel_close, hit])
    rec_reason = jnp.stack([reason_d, reason_m])
    rec_trade_direction = jnp.stack([state.direction, direction1])
    rec_entry_ts = jnp.stack([state.entry_ts, entry_ts1])
    rec_exit_ts = jnp.stack([ts_i, exit_ts])
    rec_entry_price = jnp.stack([state.entry_price, entry_price1])
    rec_exit_price = jnp.stack([exit_price_d, exit_price])
    rec_sl = jnp.stack([state.sl, sl1])
    rec_tp = jnp.stack([state.tp, tp1])
    rec_sl_atr_mult = jnp.stack([state.sl_atr_mult, sl_atr_mult1])
    rec_tp_r = jnp.stack([state.tp_r, tp_r1])
    rec_units = jnp.stack([state.units, units1])
    rec_pnl = jnp.stack([pnl_d, pnl_m])
    rec_r_mult = jnp.stack([r_mult_d, r_mult_m])
    rec_bars = jnp.stack([state.bars_in_trade, bars1])

    # ── (d) mark-to-market shaping (same op order as env_bracket) ──────────
    direction3 = direction2
    open_after = direction3 != 0
    bars3 = jnp.where(open_after, bars2 + 1, bars2)
    unrealized = jnp.where(
        open_after,
        (close_i - entry_price1) * units1 * direction3.astype(jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    reward = (equity2 - prev_equity) / reward_risk_unit
    reward = jnp.where(
        open_after,
        reward + (unrealized / jnp.maximum(risk1, 1e-12)) * params.mtm_weight - params.holding_penalty,
        reward,
    )

    # ── (e/f) advance, terminate / truncate ────────────────────────────────
    flat = direction3 == 0

    def _flat0(live, zero):
        return jnp.where(flat, zero, live)

    new_state = EnvState(
        i=state.i + 1,
        steps=state.steps + 1,
        equity=equity2,
        realized=realized2,
        direction=direction3,
        entry_ts=_flat0(entry_ts1, jnp.asarray(0, dtype=jnp.int64)),
        entry_price=_flat0(entry_price1, jnp.asarray(0.0, dtype=jnp.float64)),
        sl=_flat0(sl1, jnp.asarray(0.0, dtype=jnp.float64)),
        tp=_flat0(tp1, jnp.asarray(0.0, dtype=jnp.float64)),
        units=_flat0(units1, jnp.asarray(0.0, dtype=jnp.float64)),
        risk_cash=_flat0(risk1, jnp.asarray(0.0, dtype=jnp.float64)),
        tp_r=_flat0(tp_r1, jnp.asarray(0.0, dtype=jnp.float64)),
        sl_atr_mult=_flat0(sl_atr_mult1, jnp.asarray(0.0, dtype=jnp.float64)),
        bars_in_trade=_flat0(bars3, jnp.asarray(0, dtype=jnp.int32)),
    )
    t = params.close.shape[0]
    terminated = new_state.i >= t - 2
    truncated = new_state.steps >= params.max_episode_steps

    obs_live = observation(new_state, params)
    done = terminated | truncated
    obs = jnp.where(done, jnp.zeros_like(obs_live), obs_live)

    info = StepInfo(
        equity=equity2,
        realized=realized2,
        direction=direction3,
        closed=rec_closed,
        reason=rec_reason,
        trade_direction=rec_trade_direction,
        entry_ts=rec_entry_ts,
        exit_ts=rec_exit_ts,
        entry_price=rec_entry_price,
        exit_price=rec_exit_price,
        sl=rec_sl,
        tp=rec_tp,
        sl_atr_mult=rec_sl_atr_mult,
        tp_r=rec_tp_r,
        units=rec_units,
        pnl=rec_pnl,
        r_mult=rec_r_mult,
        bars_in_trade=rec_bars,
    )
    return obs, new_state, reward.astype(jnp.float64), terminated, truncated, info


def step_autoreset(rng: Array, state: EnvState, action: Array, params: EnvParams):
    """``step`` with gymnax-style auto-reset (D22): on done the state re-randomises
    and the returned obs is the new episode's first obs; ``final_observation`` in the
    info pytree carries the terminal obs the raw step emitted (zeros, matching
    ``env_bracket``'s done-obs — preserved, not "fixed", for SB3 parity)."""
    obs, new_state, reward, terminated, truncated, info = step(state, action, params)
    done = terminated | truncated
    rng, reset_rng = jax.random.split(rng)
    obs_reset, state_reset = reset(reset_rng, params)
    obs_out = jnp.where(done, obs_reset, obs)
    state_out = jax.tree_util.tree_map(
        lambda fresh, live: jnp.where(done, fresh, live), state_reset, new_state
    )
    return obs_out, state_out, reward, terminated, truncated, info, obs, rng


# ── first-touch reference + validation (B2 gate) ────────────────────────────


def stamps_from_ts(values, index: pd.Index) -> pd.DatetimeIndex:
    """Convert integer timestamps (``index.asi8`` values) back to tz-aware stamps.

    Uses the index's own resolution — pandas 3.x indexes are commonly
    ``datetime64[us]``, so a naive ns assumption would silently misread them.
    """
    return pd.to_datetime(np.asarray(values, dtype=np.int64),
                          unit=index.unit, utc=True).tz_convert(index.tz)


def _reference_first_touch(
    m1_ts: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    start_ts: int,
    end_ts: int,
    sl: float,
    tp: float,
    direction: int,
) -> tuple[int, int]:
    """Verbatim port of the env_bracket scan (searchsorted + masks + SL-first rule).

    Returns (reason, exit_ts) with REASON_NONE if no touch.  Used to validate the
    padded-window builder against the current scan (plan B2, D29/D30).
    """
    lo = int(np.searchsorted(m1_ts, start_ts, side="right"))
    hi = int(np.searchsorted(m1_ts, end_ts, side="right"))
    if hi <= lo:
        return REASON_NONE, 0
    highs = m1_high[lo:hi]
    lows = m1_low[lo:hi]
    if direction == 1:
        sl_mask = lows <= sl
        tp_mask = highs >= tp
    else:
        sl_mask = highs >= sl
        tp_mask = lows <= tp
    sl_any = bool(sl_mask.any())
    tp_any = bool(tp_mask.any())
    if not (sl_any or tp_any):
        return REASON_NONE, 0
    sl_rel = int(np.argmax(sl_mask)) if sl_any else None
    tp_rel = int(np.argmax(tp_mask)) if tp_any else None
    if sl_rel is not None and (tp_rel is None or sl_rel <= tp_rel):
        return REASON_SL, int(m1_ts[lo + sl_rel])
    return REASON_TP, int(m1_ts[lo + tp_rel])


def _padded_first_touch(
    high_row: np.ndarray,
    low_row: np.ndarray,
    ts_row: np.ndarray,
    sl: float,
    tp: float,
    direction: int,
) -> tuple[int, int]:
    """The gather formulation the JAX step uses (D30), evaluated in numpy."""
    if direction == 1:
        sl_mask = low_row <= sl
        tp_mask = high_row >= tp
    elif direction == -1:
        sl_mask = high_row >= sl
        tp_mask = low_row <= tp
    else:
        return REASON_NONE, 0
    sl_any = bool(sl_mask.any())
    tp_any = bool(tp_mask.any())
    if not (sl_any or tp_any):
        return REASON_NONE, 0
    sl_rel = int(np.argmax(sl_mask))
    tp_rel = int(np.argmax(tp_mask))
    if sl_any and (not tp_any or sl_rel <= tp_rel):
        return REASON_SL, int(ts_row[sl_rel])
    return REASON_TP, int(ts_row[tp_rel])


def validate_windows(
    params: EnvParams,
    m1_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    feature_cols: list[str],
    n_samples: int = 500,
    seed: int = 0,
) -> dict:
    """Randomised differential check: padded gather vs reference scan (plan B2).

    Thresholds are sampled around each bar's close ± a few ATR so both touches
    and near-misses occur.  Raises AssertionError on any mismatch; returns stats.
    """
    decision_df = decision_df.dropna(subset=list(feature_cols) + ["atr"])
    m1_ts = np.ascontiguousarray(m1_df.index.asi8.astype(np.int64))
    m1_high = np.ascontiguousarray(m1_df["High"].to_numpy(dtype=np.float64))
    m1_low = np.ascontiguousarray(m1_df["Low"].to_numpy(dtype=np.float64))
    close = decision_df["Close"].to_numpy(dtype=np.float64)
    atr = decision_df["atr"].to_numpy(dtype=np.float64)
    decision_ts = decision_df.index.asi8.astype(np.int64)

    high_pad = np.asarray(params.high_pad)
    low_pad = np.asarray(params.low_pad)
    ts_pad = np.asarray(params.ts_pad)

    rng = np.random.default_rng(seed)
    t = len(decision_df)
    n_touch = 0
    for _ in range(n_samples):
        i = int(rng.integers(0, max(t - 1, 1)))
        k = float(rng.normal(0.0, 3.0))
        sl = float(close[i] + k * max(atr[i], 1e-12))
        tp = float(close[i] - k * max(atr[i], 1e-12))
        direction = int(rng.choice([-1, 1]))
        ref = _reference_first_touch(
            m1_ts, m1_high, m1_low,
            int(decision_ts[i]),
            int(decision_ts[min(i + 1, t - 1)]),
            sl, tp, direction,
        )
        got = _padded_first_touch(high_pad[i], low_pad[i], ts_pad[i], sl, tp, direction)
        assert ref == got, f"window {i}: reference {ref} != padded {got}"
        if ref[0] != REASON_NONE:
            n_touch += 1
    return {"n_samples": n_samples, "n_touch": n_touch}
