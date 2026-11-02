from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError("Install gymnasium first: pip install gymnasium") from exc


@dataclass
class Position:
    direction: int = 0  # +1 long, -1 short, 0 flat
    entry_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    units: float = 0.0
    risk_cash: float = 0.0
    sl_distance: float = 0.0
    tp_r: float = 0.0          # planned TP R-multiple (bracket choice)
    sl_atr_mult: float = 0.0   # planned SL ATR multiplier (bracket choice)
    bars_in_trade: int = 0


class BracketTradingEnv(gym.Env):
    """Gymnasium bracket-order trading environment.

    One step = one decision bar (e.g. M5). TP/SL detection is simulated inside
    the next interval using M1 execution bars.

    Action: MultiDiscrete([3, n_sl_buckets, n_tp_buckets])
        direction: 0 flat/close, 1 long, 2 short
        sl bucket: ATR multiplier index
        tp bucket: R-multiple index

    Performance notes (Track A refactor, see DECISIONS.md D1–D8):
        * Observation inputs are precomputed as contiguous numpy arrays at
          construction time; no pandas row access happens per step.
        * The intrabar TP/SL scan is vectorised over the M1 window instead of a
          per-bar Python loop (same scan order, same SL-first tie rule).
        * Step history is kept in preallocated arrays; ``equity_curve()``
          builds the familiar DataFrame on demand.
        * ``info`` carries ``TimeLimit.truncated`` so on-policy learners
          bootstrap the value function at episode truncation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        decision_df: pd.DataFrame,
        m1_df: pd.DataFrame,
        feature_cols: List[str],
        sl_atr_multipliers=(1.0, 1.5, 2.0),
        tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
        initial_equity: float = 10_000.0,
        risk_fraction: float = 0.005,
        spread_price: float = 0.20,
        slippage_price: float = 0.02,
        commission_per_trade: float = 0.0,
        holding_penalty: float = 0.00002,
        reward_mtm_weight: float = 0.01,
        max_episode_steps: Optional[int] = None,
        randomize_start: bool = False,
    ):
        super().__init__()
        self.decision_df = decision_df.dropna(subset=feature_cols + ["atr"])
        self.feature_cols = list(feature_cols)
        self.sl_atr_multipliers = tuple(sl_atr_multipliers)
        self.tp_r_multipliers = tuple(tp_r_multipliers)
        self.initial_equity = float(initial_equity)
        self.risk_fraction = float(risk_fraction)
        self.spread_price = float(spread_price)
        self.slippage_price = float(slippage_price)
        self.commission_per_trade = float(commission_per_trade)
        self.holding_penalty = float(holding_penalty)
        self.reward_mtm_weight = float(reward_mtm_weight)
        self.max_episode_steps = max_episode_steps or (len(self.decision_df) - 2)

        self.randomize_start = randomize_start

        # ── Precomputed per-step inputs (D1/D2) ──────────────────────────────
        # Feature matrix cast once to float32 and sanitised exactly like the old
        # per-step path did (cast first, then nan_to_num) so values are bit-
        # identical to the pre-refactor implementation.
        feat32 = self.decision_df[self.feature_cols].to_numpy(dtype=np.float32)
        self._feat = np.nan_to_num(feat32, nan=0.0, posinf=10.0, neginf=-10.0)
        self._close = self.decision_df["Close"].to_numpy(dtype=np.float64)
        self._atr = self.decision_df["atr"].to_numpy(dtype=np.float64)
        self._decision_index = self.decision_df.index

        # ── M1 execution data (D6: extract arrays, no redundant frame copy) ─
        # The caller (e.g. train_ppo._slice_m1_for_decision_window) already
        # hands over a private slice; only the high/low arrays and the index are
        # used for simulation, so keep exactly one copy of those.
        self.m1_df = m1_df
        self._m1_high = np.ascontiguousarray(m1_df["High"].to_numpy(dtype=np.float64))
        self._m1_low = np.ascontiguousarray(m1_df["Low"].to_numpy(dtype=np.float64))
        self._m1_index = m1_df.index  # DatetimeIndex (sorted, tz-aware)

        self.action_space = spaces.MultiDiscrete([3, len(self.sl_atr_multipliers), len(self.tp_r_multipliers)])

        # Market features + position state.
        self.n_pos_features = 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols) + self.n_pos_features,),
            dtype=np.float32,
        )

        # ── Preallocated step history (D5) ───────────────────────────────────
        n_max = len(self.decision_df)
        self._hist_equity = np.empty(n_max, dtype=np.float64)
        self._hist_realized = np.empty(n_max, dtype=np.float64)
        self._hist_position = np.empty(n_max, dtype=np.int64)
        self._hist_close = np.empty(n_max, dtype=np.float64)
        self._hist_reward = np.empty(n_max, dtype=np.float64)
        self._hist_start = 0
        self._hist_n = 0

        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.randomize_start:
            # Random start so each episode covers a different segment of history.
            # Reserve enough room for at least max_episode_steps remaining bars.
            headroom = max(self.max_episode_steps + 2, 2)
            max_start = max(len(self.decision_df) - headroom, 1)
            self.i = int(self.np_random.integers(0, max_start))
        else:
            self.i = 0
        self.steps = 0
        self.equity = self.initial_equity
        self.realized_pnl = 0.0
        self.position = Position()
        self.trades: List[Dict[str, Any]] = []
        self._hist_start = self.i
        self._hist_n = 0
        obs = self._observation()
        return obs, {}

    # ── observation / state helpers ──────────────────────────────────────────

    def _current_row(self):
        """Compatibility shim (D7): pandas row for external callers/baselines."""
        return self.decision_df.iloc[self.i]

    def _current_time(self):
        return self._decision_index[self.i]

    def _next_time(self):
        return self._decision_index[min(self.i + 1, len(self.decision_df) - 1)]

    def _position_state_features(self) -> np.ndarray:
        close = self._close[self.i]
        atr = max(self._atr[self.i], 1e-12)
        p = self.position
        if p.direction == 0:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

        unrealized = (close - p.entry_price) * p.units * p.direction
        unrealized_r = unrealized / max(p.risk_cash, 1e-12)
        dist_tp_atr = ((p.tp - close) * p.direction) / atr
        dist_sl_atr = ((close - p.sl) * p.direction) / atr
        return np.array([
            p.direction,
            unrealized_r,
            min(p.bars_in_trade / 100.0, 10.0),
            dist_tp_atr,
            dist_sl_atr,
            p.tp_r,
        ], dtype=np.float32)

    def _observation(self):
        market = self._feat[self.i]                      # pre-sanitised float32
        pos = self._position_state_features()
        obs = np.concatenate([market, pos]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    # ── trade mechanics ──────────────────────────────────────────────────────

    def _entry_price(self, close: float, direction: int) -> float:
        return close + direction * (self.spread_price / 2.0 + self.slippage_price)

    def _exit_price(self, price: float, direction: int) -> float:
        # Long exits at bid below mid; short exits at ask above mid.
        return price - direction * (self.spread_price / 2.0 + self.slippage_price)

    def _open_position(self, direction: int, sl_idx: int, tp_idx: int):
        close = self._close[self.i]
        atr = max(self._atr[self.i], 1e-12)
        sl_atr_mult = self.sl_atr_multipliers[sl_idx]
        sl_dist = max(sl_atr_mult * atr, 1e-8)
        tp_r = self.tp_r_multipliers[tp_idx]
        entry = self._entry_price(close, direction)
        sl = entry - direction * sl_dist
        tp = entry + direction * tp_r * sl_dist
        risk_cash = max(self.equity * self.risk_fraction, 1e-8)
        units = risk_cash / sl_dist

        self.position = Position(
            direction=direction,
            entry_time=self._current_time(),
            entry_price=entry,
            sl=sl,
            tp=tp,
            units=units,
            risk_cash=risk_cash,
            sl_distance=sl_dist,
            tp_r=tp_r,
            sl_atr_mult=sl_atr_mult,
            bars_in_trade=0,
        )

    def _close_position(self, exit_price_raw: float, exit_time: pd.Timestamp, reason: str) -> float:
        p = self.position
        if p.direction == 0:
            return 0.0
        exit_price = self._exit_price(exit_price_raw, p.direction)
        pnl = (exit_price - p.entry_price) * p.units * p.direction - self.commission_per_trade
        self.equity += pnl
        self.realized_pnl += pnl
        r_mult = pnl / max(p.risk_cash, 1e-12)
        self.trades.append({
            "entry_time": p.entry_time,
            "exit_time": exit_time,
            "direction": p.direction,
            "entry_price": p.entry_price,
            "exit_price": exit_price,
            "sl": p.sl,
            "tp": p.tp,
            "sl_atr_mult": p.sl_atr_mult,   # bracket choice: SL ATR multiplier
            "tp_r_bracket": p.tp_r,          # bracket choice: planned TP R-multiple
            "units": p.units,
            "pnl": pnl,
            "r_mult": r_mult,                # realized R-multiple (actual outcome)
            "bars_in_trade": p.bars_in_trade,
            "exit_reason": reason,
        })
        self.position = Position()
        return pnl

    def _simulate_m1_until_next_decision(self) -> float:
        """Advance within the next decision interval and close if TP/SL is touched.

        Vectorised form (D3): the M1 window is scanned in one numpy pass and the
        first-touch index of each level is compared.  Semantics are identical to
        the original per-bar loop:

            * windows are half-open ``(start, end]`` (searchsorted "right"),
            * within one M1 bar SL is checked before TP (pessimistic rule), so
              a same-bar double touch resolves to SL,
            * an empty window leaves the position untouched.
        """
        p = self.position
        if p.direction == 0:
            return 0.0

        start = self._current_time()
        end = self._next_time()

        # Half-open interval (start, end]: "right" excludes start, includes end.
        lo = int(self._m1_index.searchsorted(start, side="right"))
        hi = int(self._m1_index.searchsorted(end, side="right"))
        if hi <= lo:
            return 0.0

        highs = self._m1_high[lo:hi]
        lows = self._m1_low[lo:hi]
        if p.direction == 1:
            sl_mask = lows <= p.sl
            tp_mask = highs >= p.tp
        else:
            sl_mask = highs >= p.sl
            tp_mask = lows <= p.tp

        # First touch of each level inside the window (window order preserved).
        sl_any = sl_mask.any()
        tp_any = tp_mask.any()
        if not (sl_any or tp_any):
            return 0.0

        # SL wins ties (same-bar double touch) -> pessimistic rule.
        if sl_any:
            sl_rel = int(np.argmax(sl_mask))
        else:
            sl_rel = None
        if tp_any:
            tp_rel = int(np.argmax(tp_mask))
        else:
            tp_rel = None

        if sl_rel is not None and (tp_rel is None or sl_rel <= tp_rel):
            idx = lo + sl_rel
            return self._close_position(p.sl, self._m1_index[idx], "SL")
        idx = lo + tp_rel
        return self._close_position(p.tp, self._m1_index[idx], "TP")

    # ── gym API ──────────────────────────────────────────────────────────────

    def step(self, action):
        action = np.asarray(action, dtype=int)
        direction_raw, sl_idx, tp_idx = int(action[0]), int(action[1]), int(action[2])
        desired_direction = {0: 0, 1: 1, 2: -1}[direction_raw]

        prev_equity = self.equity
        reward_risk_unit = max(prev_equity * self.risk_fraction, 1e-12)
        close = self._close[self.i]

        # Handle explicit close or flip at current decision close.
        if self.position.direction != 0:
            current_dir = self.position.direction
            if desired_direction == 0:
                self._close_position(close, self._current_time(), "manual_close")
            elif desired_direction != current_dir:
                self._close_position(close, self._current_time(), "flip_close")
                self._open_position(desired_direction, sl_idx, tp_idx)

        # Fresh entry if flat and action wants exposure.
        if self.position.direction == 0 and desired_direction != 0:
            self._open_position(desired_direction, sl_idx, tp_idx)

        # Simulate TP/SL using the M1 candles inside this decision interval.
        self._simulate_m1_until_next_decision()

        # Dense mark-to-market reward after interval.
        # Uses the CURRENT bar's close (known at decision time) to avoid lookahead.
        # The primary reward is realized PnL from TP/SL hits above; this is a small
        # shaping term (weight 0.01) that gives the agent a directional signal while
        # the trade is open, without leaking any future price information.
        if self.position.direction != 0:
            self.position.bars_in_trade += 1
            current_close = self._close[self.i]
            unrealized = (current_close - self.position.entry_price) * self.position.units * self.position.direction
        else:
            unrealized = 0.0

        reward = (self.equity - prev_equity) / reward_risk_unit
        if self.position.direction != 0:
            reward += (unrealized / max(self.position.risk_cash, 1e-12)) * self.reward_mtm_weight
            reward -= self.holding_penalty

        # Preallocated history (D5); recorded before the index advances so the
        # time column matches the pre-refactor history exactly.
        n = self._hist_n
        self._hist_equity[n] = self.equity
        self._hist_realized[n] = self.realized_pnl
        self._hist_position[n] = self.position.direction
        self._hist_close[n] = close
        self._hist_reward[n] = reward
        self._hist_n = n + 1

        self.i += 1
        self.steps += 1
        terminated = self.i >= len(self.decision_df) - 2
        truncated = self.steps >= self.max_episode_steps
        obs = self._observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)
        # D8: let on-policy learners bootstrap the value function when the
        # episode was cut by the step limit rather than by the data ending.
        info = {"equity": self.equity, "n_trades": len(self.trades),
                "TimeLimit.truncated": bool(truncated)}
        return obs, float(reward), terminated, truncated, info

    # ── results access ───────────────────────────────────────────────────────

    def equity_curve(self) -> pd.DataFrame:
        """Per-step account history, same schema as before the refactor."""
        if self._hist_n == 0:
            return pd.DataFrame(columns=["time", "equity", "realized_pnl", "position", "close", "reward"])
        n = self._hist_n
        idx = self._decision_index[self._hist_start:self._hist_start + n]
        return pd.DataFrame({
            "equity": self._hist_equity[:n],
            "realized_pnl": self._hist_realized[:n],
            "position": self._hist_position[:n],
            "close": self._hist_close[:n],
            "reward": self._hist_reward[:n],
        }, index=pd.Index(idx, name="time"))

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
