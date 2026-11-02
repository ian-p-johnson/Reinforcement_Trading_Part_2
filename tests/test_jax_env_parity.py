"""JAX env parity harness — Track B2 hard gate (SCALING_PLAN.md B2).

Golden-vector parity of ``jax_env`` against the pandas ``BracketTradingEnv``:
every scenario in ``tests/scenarios.py`` is replayed through the JAX env and
compared against the frozen ``tests/golden/*`` fixtures (DECISIONS.md P1/D9)
via ``make_golden.compare`` (f32/exact ints, f64 atol=1e-9) *and* against a
live pandas run.  Includes the B2 gate items:

  (i)  random-policy episodes at several seeds (``test_random_seed_parity``),
  (ii) hand-built edge cases incl. exact level ties (``test_exact_tie_boundary``,
       D32's exactly-representable-price construction), and
  (iii) one full sliding-fold deterministic rollout vs ``_rollout_on_split``
       (``test_fold_rollout_parity`` + ``test_metric_parity_full_report``,
       interpretation per D41) — skipped when real data/models are absent.

Platform note (D33): run the suite twice — ``JAX_PLATFORMS=cpu python -m
pytest tests/test_jax_env_parity.py`` and ``JAX_PLATFORMS=cuda …`` — so both
backends are held to the same gate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conftest  # noqa: F401  (sys.path setup)

jax = pytest.importorskip("jax")
jnp = jax.numpy

import jax_env  # noqa: E402
from make_golden import compare  # noqa: E402
from scenarios import (  # noqa: E402
    SCENARIOS, Scenario, _rng_actions, build_frame, m1_slice_for, make_m1, run_trace,
)

REASON_STR = {1: "SL", 2: "TP", 3: "manual_close", 4: "flip_close"}
TRADE_COLUMNS = [
    "entry_time", "exit_time", "direction", "entry_price", "exit_price",
    "sl", "tp", "sl_atr_mult", "tp_r_bracket", "units", "pnl", "r_mult",
    "bars_in_trade", "exit_reason",
]


# ── harness ─────────────────────────────────────────────────────────────────


def params_from_scenario(s: Scenario) -> jax_env.EnvParams:
    """Same kwargs precedence as ``Scenario.make_env`` (randomize_start is a
    reset-time concern only — start indices come from the reference run)."""
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
        max_episode_steps=None,
    )
    kw.update(s.env_kwargs)
    kw.pop("randomize_start", None)
    return jax_env.build_env_params(
        s.decision_df, m1_slice_for(s.m1, s.decision_df), s.feature_cols, **kw
    )


def _trade_rows(infos, index) -> list[dict]:
    rows = []
    for info in infos:
        for slot in (0, 1):
            if bool(info.closed[slot]):
                rows.append({
                    "entry_time": jax_env.stamps_from_ts([int(info.entry_ts[slot])], index)[0],
                    "exit_time": jax_env.stamps_from_ts([int(info.exit_ts[slot])], index)[0],
                    "direction": int(info.trade_direction[slot]),
                    "entry_price": float(info.entry_price[slot]),
                    "exit_price": float(info.exit_price[slot]),
                    "sl": float(info.sl[slot]),
                    "tp": float(info.tp[slot]),
                    "sl_atr_mult": float(info.sl_atr_mult[slot]),
                    "tp_r_bracket": float(info.tp_r[slot]),
                    "units": float(info.units[slot]),
                    "pnl": float(info.pnl[slot]),
                    "r_mult": float(info.r_mult[slot]),
                    "bars_in_trade": int(info.bars_in_trade[slot]),
                    "exit_reason": REASON_STR[int(info.reason[slot])],
                })
    return rows


def run_trace_jax(s: Scenario, start_i: int):
    """Mirror of ``scenarios.run_trace`` over the JAX env, incl. trade log.

    Start index is taken from the reference run (the env's numpy RNG stream is
    the trace's source of truth; the JAX ``reset`` stream is only used in
    training rollouts).
    """
    params = params_from_scenario(s)
    state = jax_env.init_state(params, int(start_i))

    obs_l = [np.asarray(jax_env.observation(state, params), dtype=np.float32)]
    rew_l, term_l, trunc_l = [], [], []
    eq_l, real_l, dir_l = [], [], []
    hist_eq, hist_real, hist_pos, hist_close, hist_rew, hist_time = [], [], [], [], [], []
    infos = []

    for action in s.actions:
        pre_i = int(state.i)
        close_pre = float(params.close[pre_i])
        ts_pre = int(params.decision_ts[pre_i])
        obs, state, reward, terminated, truncated, info = jax_env.step(
            state, jnp.asarray(action, dtype=jnp.int32), params
        )
        obs_l.append(np.asarray(obs, dtype=np.float32))
        rew_l.append(float(reward))
        term_l.append(bool(terminated))
        trunc_l.append(bool(truncated))
        eq_l.append(float(info.equity))
        real_l.append(float(info.realized))
        dir_l.append(int(info.direction))
        hist_eq.append(float(info.equity))
        hist_real.append(float(info.realized))
        hist_pos.append(int(info.direction))
        hist_close.append(close_pre)
        hist_rew.append(float(reward))
        hist_time.append(ts_pre)
        infos.append(info)
        if terminated or truncated:
            break

    trades = pd.DataFrame(_trade_rows(infos, s.decision_df.index), columns=TRADE_COLUMNS)
    trace = {
        "obs": np.stack(obs_l),
        "reward": np.asarray(rew_l, dtype=np.float64),
        "terminated": np.asarray(term_l, dtype=bool),
        "truncated": np.asarray(trunc_l, dtype=bool),
        "equity": np.asarray(eq_l, dtype=np.float64),
        "realized_pnl": np.asarray(real_l, dtype=np.float64),
        "direction": np.asarray(dir_l, dtype=np.int8),
        "start_i": np.asarray(int(start_i)),
        "hist_equity": np.asarray(hist_eq, dtype=np.float64),
        "hist_realized": np.asarray(hist_real, dtype=np.float64),
        "hist_close": np.asarray(hist_close, dtype=np.float64),
        "hist_reward": np.asarray(hist_rew, dtype=np.float64),
        "hist_position": np.asarray(hist_pos, dtype=np.int8),
        "hist_time_ns": np.asarray(hist_time, dtype=np.int64),
    }
    return trace, trades


# ── (gate) golden-vector parity for every scenario ──────────────────────────


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_golden_parity_jax(scenario):
    ref_trace, _ref_trades = run_trace(scenario)
    trace, trades = run_trace_jax(scenario, start_i=int(ref_trace["start_i"]))
    problems = compare(scenario.name, trace, trades)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_live_pandas_parity_jax(scenario):
    """Deep comparison against the live pandas run (tz-aware trade times)."""
    ref_trace, ref_trades = run_trace(scenario)
    trace, trades = run_trace_jax(scenario, start_i=int(ref_trace["start_i"]))
    assert len(ref_trades) == len(trades)
    for key in ("obs", "reward", "terminated", "truncated", "equity",
                "realized_pnl", "direction"):
        a, b = ref_trace[key], trace[key]
        assert a.shape == b.shape, key
        if a.dtype == np.float32:
            np.testing.assert_array_equal(a, b, err_msg=key)
        elif np.issubdtype(a.dtype, np.floating):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, err_msg=key)
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)
    if len(ref_trades):
        for col in TRADE_COLUMNS:
            a, b = ref_trades[col].to_numpy(), trades[col].to_numpy()
            if np.issubdtype(a.dtype, np.floating):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, err_msg=col)
            else:
                np.testing.assert_array_equal(a.astype(str), b.astype(str), err_msg=col)


# ── (gate i) random-policy episodes at several seeds ────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_seed_parity(seed):
    m1 = make_m1(n_days=30, seed=7)
    feat, cols = build_frame(m1)
    s = Scenario(
        name=f"random_seed{seed}",
        m1=m1, decision_df=feat, feature_cols=cols,
        actions=_rng_actions(400, seed=seed),
        env_kwargs=dict(max_episode_steps=400),
    )
    ref_trace, ref_trades = run_trace(s)
    trace, trades = run_trace_jax(s, start_i=int(ref_trace["start_i"]))
    assert trades["exit_reason"].tolist() == ref_trades["exit_reason"].tolist()
    np.testing.assert_allclose(trace["equity"], ref_trace["equity"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(trace["reward"], ref_trace["reward"], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(trace["terminated"], ref_trace["terminated"])


# ── (gate ii) exact level ties (D32) ────────────────────────────────────────


def _tie_scenario(kind: str):
    """Craft an M1 bar whose High/Low sit exactly on the bracket levels.

    Prices are exactly representable (spread/2+slippage = 0.125, ATR multiple
    and R-multiple = 1) so no FP rounding — including XLA FMA fusion — can
    alter the branch outcome.  ``kind`` selects which level the bar touches:
    ``"sl"``, ``"tp"``, or ``"both"`` (double touch must resolve to SL).
    """
    m1 = make_m1(n_days=30, seed=7)
    feat, cols = build_frame(m1)

    close0 = float(feat["Close"].iloc[0])
    atr0 = float(feat["atr"].iloc[0])
    half = 0.25 / 2.0 + 0.0  # 0.125, exactly representable
    entry = close0 + 1.0 * half
    sl_dist = max(1.0 * max(atr0, 1e-12), 1e-8)
    sl = entry - sl_dist
    tp = entry + 1.0 * sl_dist

    t0, t1 = feat.index[0], feat.index[1]
    lo = int(m1.index.searchsorted(t0, side="right"))
    hi = int(m1.index.searchsorted(t1, side="right"))
    assert hi > lo, "entry window has no M1 bars"
    i = lo
    m1 = m1.copy()
    if kind == "sl":
        low, high = sl, tp + 10 * sl_dist
    elif kind == "tp":
        low, high = entry, tp
    else:
        low, high = sl, tp
    m1.iloc[i, m1.columns.get_loc("Open")] = (low + high) / 2.0
    m1.iloc[i, m1.columns.get_loc("Close")] = (low + high) / 2.0
    m1.iloc[i, m1.columns.get_loc("High")] = high
    m1.iloc[i, m1.columns.get_loc("Low")] = low

    s = Scenario(
        name=f"tie_{kind}",
        m1=m1, decision_df=feat, feature_cols=cols,
        actions=np.array([[1, 0, 0]]),
        env_kwargs=dict(
            sl_atr_multipliers=(1.0,), tp_r_multipliers=(1.0,),
            spread_price=0.25, slippage_price=0.0, commission_per_trade=0.0,
        ),
    )
    return s, m1.index[i]


@pytest.mark.parametrize("kind,expected", [("sl", "SL"), ("tp", "TP"), ("both", "SL")])
def test_exact_tie_boundary(kind, expected):
    s, bar_ts = _tie_scenario(kind)
    ref_trace, ref_trades = run_trace(s)
    trace, trades = run_trace_jax(s, start_i=int(ref_trace["start_i"]))

    assert len(ref_trades) == 1, f"reference produced {len(ref_trades)} trades"
    assert ref_trades["exit_reason"].iloc[0] == expected, "reference tie rule"
    assert trades["exit_reason"].tolist() == ref_trades["exit_reason"].tolist()
    # The exit must land exactly on the crafted boundary bar.
    assert int(ref_trades["exit_time"].iloc[0].value) == int(bar_ts.value)
    assert int(trades["exit_time"].iloc[0].value) == int(bar_ts.value)
    # The whole trace must be bit-identical: the tie is only preserved if both
    # engines computed the same float64 level.
    for key in ("obs", "reward", "equity", "realized_pnl", "direction"):
        np.testing.assert_array_equal(trace[key], ref_trace[key], err_msg=key)


# ── window-builder validation vs the reference scan ─────────────────────────


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_window_builder_matches_reference_scan(scenario):
    params = params_from_scenario(scenario)
    stats = jax_env.validate_windows(
        params, m1_slice_for(scenario.m1, scenario.decision_df),
        scenario.decision_df, scenario.feature_cols, n_samples=200, seed=3,
    )
    assert stats["n_samples"] == 200


# ── vmap batch step == loop step (trainer prerequisite) ─────────────────────


def test_vmap_step_matches_loop():
    m1 = make_m1(n_days=30, seed=7)
    feat, cols = build_frame(m1)
    s = Scenario(name="vmap", m1=m1, decision_df=feat, feature_cols=cols,
                 actions=_rng_actions(3, seed=9))
    params = params_from_scenario(s)

    starts = jnp.asarray([0, 5, 11], dtype=jnp.int32)
    states = jax.vmap(lambda i: jax_env.init_state(params, i))(starts)
    actions = jnp.asarray(np.stack([s.actions[0], s.actions[1], s.actions[2]]),
                          dtype=jnp.int32)

    obs_v, st_v, rew_v, term_v, trunc_v, info_v = jax.vmap(
        jax_env.step, in_axes=(0, 0, None))(states, actions, params)

    for k in range(3):
        one_state = jax.tree_util.tree_map(lambda x: x[k], states)
        obs_l, st_l, rew_l, term_l, trunc_l, info_l = jax_env.step(
            one_state, actions[k], params)
        np.testing.assert_array_equal(np.asarray(obs_v[k]), np.asarray(obs_l))
        np.testing.assert_array_equal(np.asarray(rew_v[k]), np.asarray(rew_l))
        assert bool(term_v[k]) == bool(term_l)
        assert bool(trunc_v[k]) == bool(trunc_l)
        np.testing.assert_allclose(float(info_v.equity[k]), float(info_l.equity),
                                   rtol=0, atol=1e-12)
        for field in jax_env.EnvState._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(st_v, field)[k]), np.asarray(getattr(st_l, field)),
                err_msg=field)


# ── (gate iii) full sliding-fold deterministic rollout vs _rollout_on_split ─

HAS_REAL_DATA = Path("data/XAUUSD_1m_EET.csv").exists()
HAS_FOLD_MODEL = Path("models/sliding/fold_1/best_model/best_model.zip").exists()


def _capture_fold_rollout():
    """Replicates ``train_ppo._rollout_on_split`` while capturing the actions.

    The SB3 model is loaded to CPU (single-env eval) so torch does not contend
    with JAX for VRAM when the full suite runs in one process.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from data_loader import make_sliding_folds
    from config import CFG
    import train_ppo as tp

    m1, feat, feature_cols = tp._load_decision_features()
    folds = make_sliding_folds(
        feat,
        train_years=CFG.sliding_train_years,
        val_months=CFG.sliding_val_months,
        test_months=CFG.sliding_test_months,
        step_months=CFG.sliding_step_months,
        embargo_bars=CFG.split_embargo_bars,
    )
    if not folds:
        pytest.skip("no sliding folds fit the dataset")
    _tr, _va, te = folds[0]
    from model_artifacts import resolve_sb3_model_path
    fold_dir = Path("models/sliding/fold_1")
    _, ri = tp.load_run_info(str(fold_dir))
    best = Path(ri.get("best_model_path", "")) if "best_model_path" in ri else None
    if best is not None and best.exists():
        model = PPO.load(str(resolve_sb3_model_path(best)), device="cpu")
        vecnorm_path = ri.get("best_model_vecnorm_path", ri["vecnorm_path"])
    else:
        model = PPO.load(str(resolve_sb3_model_path(Path(ri["model_path"]))), device="cpu")
        vecnorm_path = ri["vecnorm_path"]
    m1_sl = tp._slice_m1_for_decision_window(m1, te)

    raw = DummyVecEnv([
        lambda: tp._CaptureDoneWrapper(
            tp.build_env(te, m1_sl, feature_cols,
                         randomize_start=False, episode_steps=None))
    ])
    venv = VecNormalize.load(str(vecnorm_path), raw)
    venv.training = False
    venv.norm_reward = False

    obs = venv.reset()
    actions = []
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(np.asarray(action[0], dtype=np.int32))
        obs, _, done, _ = venv.step(action)

    capture = venv.venv.envs[0]
    return te, m1_sl, feature_cols, np.asarray(actions), capture.saved_equity, capture.saved_trades


def _jax_rollout(te, m1_sl, feature_cols, actions):
    """Deterministic JAX rollout: returns (equity_df, trades_df)."""
    params = jax_env.build_env_params(
        te, m1_sl, feature_cols,
        sl_atr_multipliers=(1.0, 1.5, 2.0), tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
        initial_equity=10_000.0, risk_fraction=0.005, spread_price=0.20,
        slippage_price=0.02, commission_per_trade=0.01, holding_penalty=0.00002,
        reward_mtm_weight=0.01, max_episode_steps=None,
    )
    state = jax_env.init_state(params, 0)
    eq_l, rew_l = [], []
    infos = []
    for a in actions:
        _obs, state, reward, term, trunc, info = jax_env.step(
            state, jnp.asarray(a, dtype=jnp.int32), params)
        eq_l.append(float(info.equity))
        rew_l.append(float(reward))
        infos.append(info)
        if term or trunc:
            break
    eq_df = pd.DataFrame({
        "equity": np.asarray(eq_l, dtype=np.float64),
        "reward": np.asarray(rew_l, dtype=np.float64),
    }, index=pd.Index(te.index[: len(eq_l)], name="time"))
    trades = pd.DataFrame(_trade_rows(infos, te.index), columns=TRADE_COLUMNS)
    return eq_df, trades


@pytest.mark.skipif(not (HAS_REAL_DATA and HAS_FOLD_MODEL),
                    reason="real dataset / fold model not present")
def test_fold_rollout_parity():
    te, m1_sl, feature_cols, actions, eq_ref, trades_ref = _capture_fold_rollout()
    eq_got, trades_got = _jax_rollout(te, m1_sl, feature_cols, actions)

    assert len(eq_got) == len(eq_ref)
    np.testing.assert_allclose(eq_got["equity"].to_numpy(),
                               eq_ref["equity"].to_numpy(), rtol=0, atol=1e-9)
    assert len(trades_got) == len(trades_ref)
    if len(trades_got):
        np.testing.assert_allclose(trades_got["pnl"].to_numpy(),
                                   trades_ref["pnl"].to_numpy(), rtol=0, atol=1e-9)
        np.testing.assert_array_equal(trades_got["exit_reason"].to_numpy(),
                                      trades_ref["exit_reason"].to_numpy())
        np.testing.assert_array_equal(trades_got["direction"].to_numpy(),
                                      trades_ref["direction"].to_numpy())


@pytest.mark.skipif(not (HAS_REAL_DATA and HAS_FOLD_MODEL),
                    reason="real dataset / fold model not present")
def test_metric_parity_full_report():
    """D41: same policy through both engines -> identical full_report metrics."""
    from evaluate import full_report
    from config import CFG

    te, m1_sl, feature_cols, actions, eq_ref, trades_ref = _capture_fold_rollout()
    eq_got, trades_got = _jax_rollout(te, m1_sl, feature_cols, actions)

    rep_ref = full_report(eq_ref, trades_ref, initial_equity=CFG.initial_equity,
                          periods_per_year=CFG.periods_per_year)["value"].to_dict()
    rep_got = full_report(eq_got, trades_got, initial_equity=CFG.initial_equity,
                          periods_per_year=CFG.periods_per_year)["value"].to_dict()
    for k in rep_ref:
        a, b = rep_ref[k], rep_got[k]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            np.testing.assert_allclose(float(b), float(a), rtol=1e-12, atol=1e-9, err_msg=k)
        else:
            assert str(b) == str(a), k
