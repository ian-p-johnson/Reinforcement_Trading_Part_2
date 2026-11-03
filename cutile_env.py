"""Track C — bracket env-step kernel on cuTile (C1/C2, SCALING_PLAN.md).

Formulation T (tile): one block per TILE=64 environments; per-lane work mirrors
``jax_env.step`` expression-for-expression (which is itself the verified port of
``env_bracket.step``).  The intrabar first-touch scan uses the padded-window
gather (``LUT_SPEC.md`` §3.4): one ``ct.load_advanced_indexing`` row gather +
per-row min-reduces for the first-touch index and the exit timestamp.

Numerical contract (C1 — written before the kernel; the C4 parity tests are
checked against it):

* **f64** for all money/state/reward arithmetic, **identical operation order**
  to ``jax_env.step`` (entry/exit price chains, pnl, risk clamps
  ``max(x, 1e-12)``/``max(x, 1e-8)``, MtM shaping).  Position features are
  computed in f64 and cast to f32 exactly where the reference does.
* **SL-first tie rule**: same-bar double touch resolves to SL
  (``sl_rel <= tp_rel``); half-open windows (``(t_i, t_{i+1}]``) are baked into
  the host-built pads; padding sentinels are ``-inf`` highs / ``+inf`` lows /
  ``0`` times, which can never register a touch.
* **Indices int32 / timestamps int64** (cast of int tiles via ``astype`` is
  avoided — see toolchain.md: trunci compiler bug).
* **Two trade-record slots per step** (slot 0 decision-close, slot 1 intrabar)
  matching ``env_bracket``'s double-append on flip+hit steps.
* Observation = 25 sanitised market features (host-pre-sanitised, ``nan_to_num``
  semantics of ``env_bracket``) + 6 f64→f32 position features, laid out as
  ``obs_mkt [N, 32]`` (25 real + 7 zero pad) plus six ``[N]`` f32 vectors —
  the concatenation into the canonical 31-wide obs happens at the seam.

Scope cut (per plan C1): no policy, no PPO, no normalisation in this file.

Host API:
    data = CutileData.from_env_params(jax_params)   # device arrays (cupy)
    state = init_state(data, start_i_per_env)
    step_cutile(data, state, actions)              # in-place state update
    bench_cutile(data, n_envs, n_steps)            # raw steps/s (C2)

JAX seam (C3 option 2): ``make_jax_step_fn(data)`` returns a jitted function
that launches the same kernel via ``cuda.tile.jax.cutile_call`` inside a
``jax.jit`` graph.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np

TILE = 64          # environments per block (static tile shape)
W = 64             # padded window width (power of two; real windows ≤ 60)
OBS_MKT = 32       # market-feature width incl. zero pad (25 real)
TRADE_COLS = 2     # trade record slots per step

REASON_NONE, REASON_SL, REASON_TP, REASON_MANUAL, REASON_FLIP = 0, 1, 2, 3, 4


# ── kernel ──────────────────────────────────────────────────────────────────


def _build_kernel():
    """The kernel is built in a function so the module imports on machines
    without cuda-tile (tests skip).  Kernels must live in real source files —
    the compiler reads the AST (toolchain.md)."""
    import cuda.tile as ct

    @ct.kernel
    def env_step(
        feat_p, close, atr, decision_ts, high_pad, low_pad, ts_pad,
        sl_mults, tp_mults, scalar_arr, int_arr,
        s_i, s_steps, s_equity, s_realized, s_direction, s_entry_ts,
        s_entry_price, s_sl, s_tp, s_units, s_risk_cash, s_tp_r,
        s_sl_atr_mult, s_bars,
        act_dir, act_sl, act_tp,
        obs_mkt, p_dir, p_unreal, p_bars, p_dist_tp, p_dist_sl, p_tpr,
        reward, terminated, truncated,
        t_closed, t_reason, t_trade_dir, t_entry_ts, t_exit_ts, t_entry_price,
        t_exit_price, t_sl, t_tp, t_sl_atr_mult, t_tp_r, t_units, t_pnl,
        t_r_mult, t_bars,
    ):
        bid = ct.bid(0)

        # Scalars travel as f64/i32 buffers: the kernel scalar ABI is f32
        # (toolchain.md) which breaks the f64 money contract.
        sc = ct.load(scalar_arr, 0, 8)
        spread = ct.extract(sc, (0,), (1,))
        slippage = ct.extract(sc, (1,), (1,))
        commission = ct.extract(sc, (2,), (1,))
        holding_penalty = ct.extract(sc, (3,), (1,))
        mtm_weight = ct.extract(sc, (4,), (1,))
        risk_fraction = ct.extract(sc, (5,), (1,))
        ints = ct.load(int_arr, 0, 2)
        max_episode_steps = ct.extract(ints, (0,), (1,))
        T = ct.extract(ints, (1,), (1,))

        # ── load state (SoA, one tile per field) ───────────────────────────
        i0 = ct.load(s_i, bid, TILE)
        steps0 = ct.load(s_steps, bid, TILE)
        equity0 = ct.load(s_equity, bid, TILE)
        realized0 = ct.load(s_realized, bid, TILE)
        dir0 = ct.load(s_direction, bid, TILE)
        entry_ts0 = ct.load(s_entry_ts, bid, TILE)
        entry_price0 = ct.load(s_entry_price, bid, TILE)
        sl0 = ct.load(s_sl, bid, TILE)
        tp0 = ct.load(s_tp, bid, TILE)
        units0 = ct.load(s_units, bid, TILE)
        risk0 = ct.load(s_risk_cash, bid, TILE)
        tpr0 = ct.load(s_tp_r, bid, TILE)
        slam0 = ct.load(s_sl_atr_mult, bid, TILE)
        bars0 = ct.load(s_bars, bid, TILE)
        ad = ct.load(act_dir, bid, TILE)
        asl = ct.load(act_sl, bid, TILE)
        atp = ct.load(act_tp, bid, TILE)

        sm = ct.load(sl_mults, 0, 4)
        tm = ct.load(tp_mults, 0, 4)

        # ── action mapping / close-open decision (env_bracket.step) ───────
        desired = ct.where(ad == 0, 0, ct.where(ad == 1, 1, -1))
        prev_equity = equity0
        reward_risk_unit = ct.maximum(prev_equity * risk_fraction, 1e-12)
        close_i = ct.gather(close, (i0,))
        atr_i_raw = ct.gather(atr, (i0,))
        ts_i = ct.gather(decision_ts, (i0,))
        half = spread / 2.0 + slippage

        d0 = dir0.astype(ct.float64)
        close_now = (dir0 != 0) & (desired != dir0)
        open_now = (desired != 0) & (dir0 != desired)

        # ── decision-close record math ────────────────────────────────────
        exit_price_d = close_i - d0 * half
        pnl_d = (exit_price_d - entry_price0) * units0 * d0 - commission
        r_mult_d = pnl_d / ct.maximum(risk0, 1e-12)
        equity_d = equity0 + pnl_d
        realized_d = realized0 + pnl_d
        reason_d = ct.where(desired == 0, REASON_MANUAL, REASON_FLIP)

        # ── open math (post-close equity) ─────────────────────────────────
        equity_for_open = ct.where(close_now, equity_d, equity0)
        atr_i = ct.maximum(atr_i_raw, 1e-12)
        sm0 = ct.extract(sm, (0,), (1,))
        sm1 = ct.extract(sm, (1,), (1,))
        sm2 = ct.extract(sm, (2,), (1,))
        slam_o = ct.where(asl == 0, sm0, ct.where(asl == 1, sm1, sm2))
        sl_dist = ct.maximum(slam_o * atr_i, 1e-8)
        tm0 = ct.extract(tm, (0,), (1,))
        tm1 = ct.extract(tm, (1,), (1,))
        tm2 = ct.extract(tm, (2,), (1,))
        tm3 = ct.extract(tm, (3,), (1,))
        tpr_o = ct.where(atp == 0, tm0, ct.where(atp == 1, tm1,
                         ct.where(atp == 2, tm2, tm3)))
        d_open = desired.astype(ct.float64)
        entry_o = close_i + d_open * half
        sl_o = entry_o - d_open * sl_dist
        tp_o = entry_o + d_open * tpr_o * sl_dist
        risk_o = ct.maximum(equity_for_open * risk_fraction, 1e-8)
        units_o = risk_o / sl_dist

        equity1 = ct.where(open_now, equity_for_open,
                           ct.where(close_now, equity_d, equity0))
        realized1 = ct.where(close_now, realized_d, realized0)
        dir1 = ct.where(open_now, desired, ct.where(close_now, 0, dir0))
        entry_ts1 = ct.where(open_now, ts_i, entry_ts0)
        entry_price1 = ct.where(open_now, entry_o, entry_price0)
        sl1 = ct.where(open_now, sl_o, sl0)
        tp1 = ct.where(open_now, tp_o, tp0)
        units1 = ct.where(open_now, units_o, units0)
        risk1 = ct.where(open_now, risk_o, risk0)
        tpr1 = ct.where(open_now, tpr_o, tpr0)
        slam1 = ct.where(open_now, slam_o, slam0)
        bars1 = ct.where(open_now, 0, bars0)

        # ── intrabar first-touch over the padded window ───────────────────
        row = i0
        highs = ct.load_advanced_indexing(high_pad, (row, ct.Slice(0, W)))
        lows = ct.load_advanced_indexing(low_pad, (row, ct.Slice(0, W)))
        ts_row = ct.load_advanced_indexing(ts_pad, (row, ct.Slice(0, W)))

        is_long = (dir1 == 1).reshape((TILE, 1))
        d1 = dir1.astype(ct.float64)
        open_lane = (dir1 != 0).reshape((TILE, 1))
        sl1_r = sl1.reshape((TILE, 1))
        tp1_r = tp1.reshape((TILE, 1))
        sl_touch = ct.where(is_long, lows <= sl1_r, highs >= sl1_r) & open_lane
        tp_touch = ct.where(is_long, highs >= tp1_r, lows <= tp1_r) & open_lane

        j = ct.arange(W, dtype=ct.int32)
        big = ct.full((W,), 99, ct.int32)
        sl_rel = ct.reduce(ct.where(sl_touch, j, big), axis=1,
                           func=ct.minimum, identity=99)
        tp_rel = ct.reduce(ct.where(tp_touch, j, big), axis=1,
                           func=ct.minimum, identity=99)
        sl_any = sl_rel < 99
        tp_any = tp_rel < 99
        sl_wins = sl_any & ((~tp_any) | (sl_rel <= tp_rel))
        touch = sl_any | tp_any
        close_at_sl = touch & sl_wins
        close_at_tp = touch & (~sl_wins)
        hit = close_at_sl | close_at_tp

        rel = ct.where(close_at_sl, sl_rel, tp_rel)
        # exit timestamp via in-bounds gather (the i64 one-hot min-reduce
        # miscompiles on this toolchain — toolchain.md; rel==99 sentinel maps
        # to row 0, masked out by closed=False in the record reconstruction).
        rel_safe = ct.where(rel < W, rel, 0)
        exit_ts = ct.gather(ts_pad, (row, rel_safe))
        exit_price_raw = ct.where(close_at_sl, sl1, tp1)
        exit_price = exit_price_raw - d1 * half
        pnl_m = (exit_price - entry_price1) * units1 * d1 - commission
        r_mult_m = pnl_m / ct.maximum(risk1, 1e-12)
        reason_m = ct.where(close_at_sl, REASON_SL, REASON_TP)

        equity2 = equity1 + ct.where(hit, pnl_m, 0.0)
        realized2 = realized1 + ct.where(hit, pnl_m, 0.0)
        dir2 = ct.where(hit, 0, dir1)
        bars2 = ct.where(hit, 0, bars1)

        # ── mark-to-market shaping ────────────────────────────────────────
        open_after = dir2 != 0
        d2 = dir2.astype(ct.float64)
        bars3 = ct.where(open_after, bars2 + 1, bars2)
        unrealized = ct.where(
            open_after, (close_i - entry_price1) * units1 * d2, 0.0)
        reward_v = (equity2 - prev_equity) / reward_risk_unit
        reward_v = ct.where(
            open_after,
            reward_v + (unrealized / ct.maximum(risk1, 1e-12)) * mtm_weight
            - holding_penalty,
            reward_v)

        # ── advance / flags ───────────────────────────────────────────────
        i2 = i0 + 1
        steps2 = steps0 + 1
        term_v = i2 >= (T - 2)
        trunc_v = steps2 >= max_episode_steps

        # flat-reset of position fields (Position() semantics)
        flat = dir2 == 0
        entry_ts2 = ct.where(flat, 0, entry_ts1)
        entry_price2 = ct.where(flat, 0.0, entry_price1)
        sl2 = ct.where(flat, 0.0, sl1)
        tp2 = ct.where(flat, 0.0, tp1)
        units2 = ct.where(flat, 0.0, units1)
        risk2 = ct.where(flat, 0.0, risk1)
        tpr2 = ct.where(flat, 0.0, tpr1)
        slam2 = ct.where(flat, 0.0, slam1)
        bars_final = ct.where(flat, 0, bars3)

        # ── store updated state ───────────────────────────────────────────
        ct.store(s_i, bid, i2)
        ct.store(s_steps, bid, steps2)
        ct.store(s_equity, bid, equity2)
        ct.store(s_realized, bid, realized2)
        ct.store(s_direction, bid, dir2)
        ct.store(s_entry_ts, bid, entry_ts2)
        ct.store(s_entry_price, bid, entry_price2)
        ct.store(s_sl, bid, sl2)
        ct.store(s_tp, bid, tp2)
        ct.store(s_units, bid, units2)
        ct.store(s_risk_cash, bid, risk2)
        ct.store(s_tp_r, bid, tpr2)
        ct.store(s_sl_atr_mult, bid, slam2)
        ct.store(s_bars, bid, bars_final)

        # ── observation: pos features use the NEW bar (i2) and the NEW
        # state fields, exactly like jax_env.observation(new_state) ────────
        done = term_v | trunc_v
        zero = ct.where(done, 0.0, 1.0)
        mkt = ct.load_advanced_indexing(feat_p, (i2, ct.Slice(0, OBS_MKT)))
        mkt = mkt * zero.reshape((TILE, 1))
        ct.store(obs_mkt, (bid, 0), mkt)
        close2 = ct.gather(close, (i2,))
        atr2 = ct.maximum(ct.gather(atr, (i2,)), 1e-12)
        d2o = dir2.astype(ct.float64)
        unrealized_obs = (close2 - entry_price2) * units2 * d2o
        unrealized_r = unrealized_obs / ct.maximum(risk2, 1e-12)
        dist_tp_atr = ((tp2 - close2) * d2o) / atr2
        dist_sl_atr = ((close2 - sl2) * d2o) / atr2
        bars_frac = ct.minimum(bars_final.astype(ct.float64) / 100.0, 10.0)
        ct.store(p_dir, bid, (ct.where(flat, 0.0, d2o) * zero).astype(ct.float32))
        ct.store(p_unreal, bid, (ct.where(flat, 0.0, unrealized_r) * zero).astype(ct.float32))
        ct.store(p_bars, bid, (ct.where(flat, 0.0, bars_frac) * zero).astype(ct.float32))
        ct.store(p_dist_tp, bid, (ct.where(flat, 0.0, dist_tp_atr) * zero).astype(ct.float32))
        ct.store(p_dist_sl, bid, (ct.where(flat, 0.0, dist_sl_atr) * zero).astype(ct.float32))
        ct.store(p_tpr, bid, (ct.where(flat, 0.0, tpr2) * zero).astype(ct.float32))

        # reward is NOT zeroed at done (parity with jax_env.step).
        ct.store(reward, bid, reward_v)
        ct.store(terminated, bid, term_v.astype(ct.int32))
        ct.store(truncated, bid, trunc_v.astype(ct.int32))

        # ── trade records: slot 0 = decision close, slot 1 = intrabar ─────
        j2 = ct.arange(2, dtype=ct.int32).reshape((1, 2))

        def pack2(a, b):
            return ct.where(j2 == 0, a.reshape((TILE, 1)), b.reshape((TILE, 1)))

        ct.store(t_closed, (bid, 0),
                 pack2(close_now.astype(ct.int32), hit.astype(ct.int32)))
        ct.store(t_reason, (bid, 0), pack2(reason_d, reason_m))
        ct.store(t_trade_dir, (bid, 0), pack2(dir0, dir1))
        ct.store(t_entry_ts, (bid, 0), pack2(entry_ts0, entry_ts1))
        ct.store(t_exit_ts, (bid, 0), pack2(ts_i, exit_ts))
        ct.store(t_entry_price, (bid, 0), pack2(entry_price0, entry_price1))
        ct.store(t_exit_price, (bid, 0), pack2(exit_price_d, exit_price))
        ct.store(t_sl, (bid, 0), pack2(sl0, sl1))
        ct.store(t_tp, (bid, 0), pack2(tp0, tp1))
        ct.store(t_sl_atr_mult, (bid, 0), pack2(slam0, slam1))
        ct.store(t_tp_r, (bid, 0), pack2(tpr0, tpr1))
        ct.store(t_units, (bid, 0), pack2(units0, units1))
        ct.store(t_pnl, (bid, 0), pack2(pnl_d, pnl_m))
        ct.store(t_r_mult, (bid, 0), pack2(r_mult_d, r_mult_m))
        ct.store(t_bars, (bid, 0), pack2(bars0, bars1))

    return env_step


KERNEL = None


def get_kernel():
    global KERNEL
    if KERNEL is None:
        KERNEL = _build_kernel()
    return KERNEL


# ── host side (cupy arrays for the raw harness) ─────────────────────────────


def _require_cupy():
    import cupy
    return cupy


@dataclass
class CutileData:
    """Device-resident dataset + scalars for the env-step kernel."""

    feat_p: object        # f32 [T, 32]  market features (25 real + 7 zero)
    close: object         # f64 [T]
    atr: object           # f64 [T]
    decision_ts: object   # i64 [T]
    high_pad: object      # f64 [T, 64]  padding -inf
    low_pad: object       # f64 [T, 64]  padding +inf
    ts_pad: object        # i64 [T, 64]  padding 0
    sl_mults: object      # f64 [4]
    tp_mults: object      # f64 [4]
    spread: float
    slippage: float
    commission: float
    holding_penalty: float
    mtm_weight: float
    risk_fraction: float
    initial_equity: float
    max_episode_steps: int
    T: int
    scalar_arr: object    # f64 [8] — kernel scalar ABI is f32 (toolchain.md)
    int_arr: object       # i32 [2] — [max_episode_steps, T]

    @classmethod
    def from_env_params(cls, params) -> "CutileData":
        """Convert a ``jax_env.EnvParams`` (numpy-backed) to kernel arrays."""
        cp = _require_cupy()
        feat = np.asarray(params.feat)
        t = feat.shape[0]
        feat_p = np.zeros((t, OBS_MKT), dtype=np.float32)
        feat_p[:, : feat.shape[1]] = feat

        def pad2d(x, value):
            x = np.asarray(x)
            out = np.full((x.shape[0], W), value, dtype=x.dtype)
            out[:, : x.shape[1]] = x
            return out

        sl_m = np.zeros(4, dtype=np.float64)
        sl_m[: len(params.sl_mults)] = np.asarray(params.sl_mults)
        tp_m = np.zeros(4, dtype=np.float64)
        tp_m[: len(params.tp_mults)] = np.asarray(params.tp_mults)

        return cls(
            feat_p=cp.asarray(feat_p),
            close=cp.asarray(np.asarray(params.close)),
            atr=cp.asarray(np.asarray(params.atr)),
            decision_ts=cp.asarray(np.asarray(params.decision_ts)),
            high_pad=cp.asarray(pad2d(params.high_pad, -np.inf)),
            low_pad=cp.asarray(pad2d(params.low_pad, np.inf)),
            ts_pad=cp.asarray(pad2d(params.ts_pad, 0)),
            sl_mults=cp.asarray(sl_m),
            tp_mults=cp.asarray(tp_m),
            spread=float(params.spread),
            slippage=float(params.slippage),
            commission=float(params.commission),
            holding_penalty=float(params.holding_penalty),
            mtm_weight=float(params.mtm_weight),
            risk_fraction=float(params.risk_fraction),
            initial_equity=float(params.initial_equity),
            max_episode_steps=int(params.max_episode_steps),
            T=t,
            scalar_arr=cp.asarray(np.array([
                float(params.spread), float(params.slippage),
                float(params.commission), float(params.holding_penalty),
                float(params.mtm_weight), float(params.risk_fraction),
                float(params.max_episode_steps), float(t)], dtype=np.float64)),
            int_arr=cp.asarray(np.array([
                int(params.max_episode_steps), t], dtype=np.int32)),
        )


STATE_FIELDS = (
    "i", "steps", "equity", "realized", "direction", "entry_ts",
    "entry_price", "sl", "tp", "units", "risk_cash", "tp_r",
    "sl_atr_mult", "bars",
)


class CutileState(NamedTuple):
    i: object
    steps: object
    equity: object
    realized: object
    direction: object
    entry_ts: object
    entry_price: object
    sl: object
    tp: object
    units: object
    risk_cash: object
    tp_r: object
    sl_atr_mult: object
    bars: object


def init_state(data: CutileData, start_i) -> CutileState:
    """Flat-position state at given start indices (mirrors jax_env.init_state)."""
    cp = _require_cupy()
    n = len(start_i)
    start = cp.asarray(np.asarray(start_i, dtype=np.int32))
    zf = lambda: cp.zeros(n, dtype=cp.float64)
    return CutileState(
        i=start,
        steps=cp.zeros(n, dtype=cp.int32),
        equity=cp.full(n, data.initial_equity, dtype=cp.float64),
        realized=zf(),
        direction=cp.zeros(n, dtype=cp.int32),
        entry_ts=cp.zeros(n, dtype=cp.int64),
        entry_price=zf(), sl=zf(), tp=zf(), units=zf(), risk_cash=zf(),
        tp_r=zf(), sl_atr_mult=zf(),
        bars=cp.zeros(n, dtype=cp.int32),
    )


TRADE_FIELDS = (
    "closed", "reason", "trade_dir", "entry_ts", "exit_ts", "entry_price",
    "exit_price", "sl", "tp", "sl_atr_mult", "tp_r", "units", "pnl",
    "r_mult", "bars",
)


class StepOutputs(NamedTuple):
    obs_mkt: object        # f32 [N, 32]
    p_dir: object          # f32 [N] (5 more pos vectors)
    p_unreal: object
    p_bars: object
    p_dist_tp: object
    p_dist_sl: object
    p_tpr: object
    reward: object         # f64 [N]
    terminated: object     # i32 [N]
    truncated: object      # i32 [N]
    trades: dict           # field -> array [N, 2]


def allocate_outputs(n: int) -> StepOutputs:
    cp = _require_cupy()
    zf = lambda: cp.zeros(n, dtype=cp.float64)
    trades = {
        "closed": cp.zeros((n, 2), dtype=cp.int32),
        "reason": cp.zeros((n, 2), dtype=cp.int32),
        "trade_dir": cp.zeros((n, 2), dtype=cp.int32),
        "entry_ts": cp.zeros((n, 2), dtype=cp.int64),
        "exit_ts": cp.zeros((n, 2), dtype=cp.int64),
    }
    for f in ("entry_price", "exit_price", "sl", "tp", "sl_atr_mult", "tp_r",
              "units", "pnl", "r_mult"):
        trades[f] = cp.zeros((n, 2), dtype=cp.float64)
    trades["bars"] = cp.zeros((n, 2), dtype=cp.int32)
    return StepOutputs(
        obs_mkt=cp.zeros((n, OBS_MKT), dtype=cp.float32),
        p_dir=cp.zeros(n, dtype=cp.float32),
        p_unreal=cp.zeros(n, dtype=cp.float32),
        p_bars=cp.zeros(n, dtype=cp.float32),
        p_dist_tp=cp.zeros(n, dtype=cp.float32),
        p_dist_sl=cp.zeros(n, dtype=cp.float32),
        p_tpr=cp.zeros(n, dtype=cp.float32),
        reward=zf(),
        terminated=cp.zeros(n, dtype=cp.int32),
        truncated=cp.zeros(n, dtype=cp.int32),
        trades=trades,
    )


def _kernel_args(data: CutileData, state: CutileState, actions, out: StepOutputs):
    act_dir, act_sl, act_tp = actions
    t = out.trades
    return (
        data.feat_p, data.close, data.atr, data.decision_ts,
        data.high_pad, data.low_pad, data.ts_pad,
        data.sl_mults, data.tp_mults, data.scalar_arr, data.int_arr,
        state.i, state.steps, state.equity, state.realized, state.direction,
        state.entry_ts, state.entry_price, state.sl, state.tp, state.units,
        state.risk_cash, state.tp_r, state.sl_atr_mult, state.bars,
        act_dir, act_sl, act_tp,
        out.obs_mkt, out.p_dir, out.p_unreal, out.p_bars, out.p_dist_tp,
        out.p_dist_sl, out.p_tpr, out.reward, out.terminated, out.truncated,
        t["closed"], t["reason"], t["trade_dir"], t["entry_ts"], t["exit_ts"],
        t["entry_price"], t["exit_price"], t["sl"], t["tp"], t["sl_atr_mult"],
        t["tp_r"], t["units"], t["pnl"], t["r_mult"], t["bars"],
    )


def step_cutile(data: CutileData, state: CutileState, actions,
                out: StepOutputs | None = None, stream=None) -> StepOutputs:
    """Launch one env step for all N envs; state arrays update in place."""
    cp = _require_cupy()
    import cuda.tile as ct

    n = int(state.i.shape[0])
    if n % TILE != 0:
        raise ValueError(f"N must be a multiple of {TILE}, got {n}")
    if out is None:
        out = allocate_outputs(n)
    if stream is None:
        stream = cp.cuda.get_current_stream()
    acts = tuple(cp.asarray(a) for a in actions)
    kernel = get_kernel()
    ct.launch(stream, (ct.cdiv(n, TILE),), kernel, _kernel_args(data, state, acts, out))
    return out


def bench_cutile(data: CutileData, n_envs: int, n_steps: int,
                 seed: int = 0, stream=None) -> dict:
    """Raw kernel throughput (C2): n_steps sequential launches over n_envs.

    State persists across launches (no host round-trip).  Includes per-launch
    host overhead — the honest number for the per-step launch contract.
    """
    cp = _require_cupy()
    rng = np.random.default_rng(seed)
    state = init_state(data, np.zeros(n_envs, dtype=np.int32))
    out = allocate_outputs(n_envs)
    if stream is None:
        stream = cp.cuda.get_current_stream()
    actions = (
        cp.asarray(rng.integers(0, 3, size=n_envs).astype(np.int32)),
        cp.asarray(rng.integers(0, 3, size=n_envs).astype(np.int32)),
        cp.asarray(rng.integers(0, 4, size=n_envs).astype(np.int32)),
    )
    for _ in range(3):                       # warmup / JIT
        step_cutile(data, state, actions, out, stream)
    stream.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step_cutile(data, state, actions, out, stream)
    stream.synchronize()
    dt = time.perf_counter() - t0
    return {"n_envs": n_envs, "n_steps": n_steps, "seconds": dt,
            "steps_per_s": n_envs * n_steps / dt,
            "launches_per_s": n_steps / dt}


# ── JAX seam (C3 option 2): the kernel inside jax.jit via cutile_call ──────


def make_jax_step_fn():
    """Return ``step(inputs...) -> outputs`` jitted through the cuTile JAX FFI.

    All buffers are jax arrays; state is threaded through ``InputOutput``
    aliases so the kernel updates it in place inside the graph.  This is the
    C3 option-2 seam: the JAX trainer keeps its rollout/GAE/update while the
    env step becomes a single custom op.
    """
    import jax
    import jax.numpy as jnp
    from cuda.tile.jax import OutputPlaceholder, cutile_call, InputOutput

    kernel = get_kernel()
    n_static = None  # inferred per call

    def step(feat_p, close, atr, decision_ts, high_pad, low_pad, ts_pad,
             sl_mults, tp_mults, scalar_arr, int_arr, state, actions):
        n = state[0].shape[0]
        grid = ((n + TILE - 1) // TILE,)
        s_i, s_steps, s_equity, s_realized, s_direction, s_entry_ts, \
            s_entry_price, s_sl, s_tp, s_units, s_risk_cash, s_tp_r, \
            s_sl_atr_mult, s_bars = state
        act_dir, act_sl, act_tp = actions

        obs_mkt_ph = OutputPlaceholder((n, OBS_MKT), jnp.float32)
        p_ph = [OutputPlaceholder((n,), jnp.float32) for _ in range(6)]
        reward_ph = OutputPlaceholder((n,), jnp.float64)
        term_ph = OutputPlaceholder((n,), jnp.int32)
        trunc_ph = OutputPlaceholder((n,), jnp.int32)
        t_int_ph = [OutputPlaceholder((n, 2), jnp.int32) for _ in range(3)]
        t_i64_ph = [OutputPlaceholder((n, 2), jnp.int64) for _ in range(2)]
        t_f64_ph = [OutputPlaceholder((n, 2), jnp.float64) for _ in range(9)]
        t_bars_ph = OutputPlaceholder((n, 2), jnp.int32)

        args = (
            feat_p, close, atr, decision_ts, high_pad, low_pad, ts_pad,
            sl_mults, tp_mults, scalar_arr, int_arr,
            InputOutput(s_i), InputOutput(s_steps), InputOutput(s_equity),
            InputOutput(s_realized), InputOutput(s_direction),
            InputOutput(s_entry_ts), InputOutput(s_entry_price), InputOutput(s_sl),
            InputOutput(s_tp), InputOutput(s_units), InputOutput(s_risk_cash),
            InputOutput(s_tp_r), InputOutput(s_sl_atr_mult), InputOutput(s_bars),
            act_dir, act_sl, act_tp,
            obs_mkt_ph, *p_ph, reward_ph, term_ph, trunc_ph,
            *t_int_ph, *t_i64_ph, *t_f64_ph, t_bars_ph,
        )
        outs = cutile_call(grid, kernel, args)
        return outs

    return jax.jit(step)


# ── C2 raw harness entry point ──────────────────────────────────────────────


def main() -> int:
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    import jax_env

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", type=int, nargs="*", default=[1024, 4096, 16384, 65536])
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    from scenarios import build_frame, make_m1, m1_slice_for
    m1 = make_m1(n_days=30, seed=7)
    feat, cols = build_frame(m1)
    params = jax_env.build_env_params(
        feat, m1_slice_for(m1, feat), cols, commission_per_trade=0.01,
        max_episode_steps=2048)
    data = CutileData.from_env_params(params)
    print(f"dataset: T={data.T}, W={W}, tile={TILE}")
    for n in args.widths:
        r = bench_cutile(data, n, args.steps)
        print(f"[cutile ] env step @ {n:,} envs: {r['steps_per_s']:,.0f} steps/s "
              f"({r['launches_per_s']:,.0f} launches/s, {r['seconds']:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def bench_seam(n_envs: int = 4096, n_steps: int = 128, seed: int = 0) -> dict:
    """C3 measurement: identical K-step rollouts over N envs, three ways —

    (a) the cuTile kernel through the JAX FFI inside ``lax.scan``,
    (b) ``jax_env.step`` vmapped inside ``lax.scan`` (the Track-B rollout env),
    both excluding policy/GAE/update, plus (c) the raw cupy launch loop.
    """
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent / "tests"))
    import jax
    import jax.numpy as jnp
    import jax_env as je
    from scenarios import build_frame, make_m1, m1_slice_for

    m1 = make_m1(n_days=30, seed=7)
    feat, cols = build_frame(m1)
    params = je.build_env_params(feat, m1_slice_for(m1, feat), cols,
                                 commission_per_trade=0.01, max_episode_steps=2048)
    data = CutileData.from_env_params(params)

    rng = np.random.default_rng(seed)
    acts_np = (rng.integers(0, 3, size=(n_steps, n_envs)).astype(np.int32),
               rng.integers(0, 3, size=(n_steps, n_envs)).astype(np.int32),
               rng.integers(0, 4, size=(n_steps, n_envs)).astype(np.int32))
    to_jax = lambda x: jnp.asarray(np.asarray(x.get() if hasattr(x, "get") else x))
    data_args = tuple(to_jax(getattr(data, f)) for f in (
        "feat_p", "close", "atr", "decision_ts", "high_pad", "low_pad",
        "ts_pad", "sl_mults", "tp_mults", "scalar_arr", "int_arr"))
    step_fn = make_jax_step_fn()
    n_state = len(STATE_FIELDS)

    def scan_seam(state_args, acts):
        def body(st, a):
            outs = step_fn(*data_args, st, a)
            return tuple(outs[:n_state]), outs[n_state + 7]  # new state, reward
        return jax.lax.scan(body, state_args, acts)

    def scan_jaxenv(state_pytree, acts):
        def body(st, a):
            def one(s, ai):
                _o, s2, r, _t, _tr, _i = je.step(s, ai, params)
                return s2, r
            st2, r = jax.vmap(one)(st, jnp.stack(a, axis=-1))
            return st2, r
        return jax.lax.scan(body, state_pytree, acts)

    state0 = init_state(data, np.zeros(n_envs, dtype=np.int32))
    state_args = tuple(to_jax(getattr(state0, f)) for f in STATE_FIELDS)
    acts_j = tuple(to_jax(a) for a in acts_np)

    scan_seam_j = jax.jit(scan_seam)
    t0 = time.perf_counter()
    outs = scan_seam_j(state_args, acts_j)
    jax.block_until_ready(outs)
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    outs = scan_seam_j(state_args, acts_j)
    jax.block_until_ready(outs)
    dt_seam = time.perf_counter() - t0

    state0_p = jax.vmap(lambda i: je.init_state(params, i))(
        jnp.zeros((n_envs,), dtype=jnp.int32))
    scan_jj = jax.jit(scan_jaxenv)
    t0 = time.perf_counter()
    outs2 = scan_jj(state0_p, acts_j)
    jax.block_until_ready(outs2)
    t_compile2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    outs2 = scan_jj(state0_p, acts_j)
    jax.block_until_ready(outs2)
    dt_jax = time.perf_counter() - t0

    total = n_envs * n_steps
    return {
        "seam_cutile_steps_per_s": total / dt_seam,
        "jaxenv_vmap_steps_per_s": total / dt_jax,
        "seam_compile_s": t_compile,
        "jaxenv_compile_s": t_compile2,
    }
