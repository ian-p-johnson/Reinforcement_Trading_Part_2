from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ── Timeframe helpers ────────────────────────────────────────────────────────

# Map every common notation (MT4/MT5, shorthand, pandas) to the canonical
# pandas resample rule.  Add rows here to support new timeframes.
_TF_TO_PANDAS: dict[str, str] = {
    # MT4/MT5     shorthand    pandas rule
    "M1":  "1min",  "1m":  "1min",  "1min":  "1min",
    "M5":  "5min",  "5m":  "5min",  "5min":  "5min",
    "M15": "15min", "15m": "15min", "15min": "15min",
    "M30": "30min", "30m": "30min", "30min": "30min",
    "H1":  "1h",    "1h":  "1h",   "1H":    "1h",
    "H4":  "4h",    "4h":  "4h",   "4H":    "4h",
    "D1":  "1D",    "1d":  "1D",   "1D":    "1D",
}

# Approximate trading bars per year for XAUUSD.
# Basis: 23 trading hours/day × 261 trading days/year
# (XAUUSD trades ~Sun 22:00 UTC to Fri 22:00 UTC, minus ~1h daily maintenance)
_BARS_PER_YEAR: dict[str, int] = {
    "1min":  360_180,   # 23 × 60 × 261
    "5min":   72_036,   # 23 × 12 × 261
    "15min":  24_012,   # 23 × 4  × 261
    "30min":  12_006,   # 23 × 2  × 261
    "1h":      6_003,   # 23 × 1  × 261
    "4h":      1_501,   # 23/4    × 261  (≈ 6 bars/day)
    "1D":        261,   # 1       × 261
}


@dataclass
class ProjectConfig:
    # Path to your full MT4/MT5-style minute CSV.
    # Long Bid file (May 2003 – May 2026, ~23 years) — primary dataset: enough
    # history for several walk-forward folds plus a multi-year sealed test.
    csv_path: Path = Path("data/XAUUSD_1m_EET.csv")
    # Short Ask file (Jan 2020 – Jan 2026, ~6 years) — kept as a fast smoke-test
    # dataset; uncomment to fall back to it.
    #csv_path: Path = Path("data/XAUUSD_1 Min_Ask_2020.01.09_2026.01.15.csv")

    # Your file column is usually exactly "Time (EET)".
    time_col: str = "Time (EET)"

    # Many brokers call this EET but actually follow EET/EEST server time.
    # Europe/Helsinki is a practical EET/EEST timezone choice.
    source_tz: str = "Europe/Helsinki"

    # MT4/MT5-style M1 exports usually timestamp each row at candle OPEN.
    # Internally the project indexes bars by CLOSE time so a decision timestamp
    # means "this candle is complete and known".
    timestamp_is_bar_open: bool = True

    # Execution stays at M1 for realistic TP/SL fill simulation.
    # decision_timeframe controls the bar at which the RL agent observes and
    # acts.  Use any key from _TF_TO_PANDAS: "M5", "H1", "H4", "D1", etc.
    execution_timeframe: str = "1min"
    decision_timeframe: str = "H1"

    # DEMO MODE: set to None for a full production run.
    # 90 days ≈ 25 K M5 bars / ~2 K H1 bars. Enough for pipeline smoke-tests
    # but too small for a publishable result.
    max_days_for_demo: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # ── Validation scheme ────────────────────────────────────────────────────
    # Two compatible modes share the SAME sealed final-test segment:
    #
    #   (a) Single chronological split  → data_loader.split_train_val_test()
    #       train = first train_frac, val = next val_frac, test = remaining.
    #       Used by run_pipeline.py / view_results.py / final_holdout_eval.py.
    #
    #   (b) Walk-forward (default for RL) → data_loader.make_walk_forward_folds()
    #       The last test_frac of bars is sealed as the final holdout; the
    #       remaining "development" region is cut into (n_folds + 1) equal blocks
    #       whose validation windows roll forward in time.  See train_ppo.train_walk_forward().
    #
    # test_frac is kept equal to (1 - train_frac - val_frac) so the sealed test
    # is byte-for-byte identical in both modes — whichever one produced the model,
    # final_holdout_eval.py evaluates it on the same untouched holdout.
    #
    # Long Bid dataset (May 2003 – May 2026, ~23 years of H1 ≈ 138 K bars):
    #   • single split → ~16 yr train / ~3.5 yr val / ~3.5 yr test
    #   • walk-forward → 5 folds rolling over the first ~19.5 yr, last ~3.5 yr sealed
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1        # sealed final holdout (== 1 - train_frac - val_frac)

    # Walk-forward validation (block scheme — train_ppo.train_walk_forward).
    n_walk_forward_folds: int = 5       # number of rolling (train → val) folds
    walk_forward_anchored: bool = False  # True = expanding train window; False = rolling fixed-size

    # ── Sliding-window walk-forward (realistic retrain simulation) ───────────
    # train_ppo.train_sliding_walk_forward(): each fold trains on
    # `sliding_train_years`, selects its checkpoint on the next
    # `sliding_val_months`, then is judged TRUE out-of-sample on the following
    # `sliding_test_months`. The whole window then slides forward by
    # `sliding_step_months` and repeats — simulating "retrain every step_months,
    # then trade the next test_months live." Stitching every fold's test window
    # gives one continuous out-of-sample equity track record.
    #
    # Because every train window is the SAME calendar length, equal timesteps per
    # fold already means equal passes over the data (no per-fold scaling needed).
    #
    # Note on cost: with a 6-month step over ~23 years you get ~34 folds, i.e.
    # ~34 model trainings. Raise sliding_step_months (e.g. 12) to halve the count.
    sliding_train_years: float = 5.0
    sliding_val_months: int = 6
    sliding_test_months: int = 6
    sliding_step_months: int = 6

    # ── Walk-forward deployment gate ─────────────────────────────────────────
    # After the folds finish, the final fold is promoted to the production slot
    # (models/) ONLY if the out-of-sample folds are consistently good. Otherwise
    # nothing ships: the artifacts stay quarantined under models/walk_forward/
    # and any stale models/run_info.json is set aside as run_info.NO_DEPLOY.json.
    # The gate only ever PREVENTS a bad deploy — it never fabricates one.
    min_consistent_folds: int = 4              # folds needing return>0 AND PF>gate_min_profit_factor
    gate_min_profit_factor: float = 1.0        # a fold "passes" when its val PF exceeds this
    gate_worst_fold_min_pf: float = 0.9        # AND no single fold's val PF may fall below this
    gate_require_mean_sharpe_positive: bool = True  # AND mean val Sharpe across folds must be > 0

    # Embargo removes bars on each side of every split boundary.
    # EMA-200 (the longest lookback used) retains ~37% of a bar's weight after
    # 200 steps; 200-bar embargo makes cross-boundary leakage negligible.
    split_embargo_bars: int = 200

    # Indicators/features.
    atr_period: int = 14
    rsi_period: int = 14
    warmup_bars: int = 250

    # Bracket choices for the RL action space.
    sl_atr_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0)
    tp_r_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

    # Backtest/account model.
    initial_equity: float = 10_000.0
    risk_fraction: float = 0.005  # 0.5% equity risked per trade.
    spread_price: float = 0.20    # XAUUSD price units; adjust to your broker.
    slippage_price: float = 0.02  # XAUUSD price units per side.
    commission_per_trade: float = 0.01

    # Reward shaping.
    holding_penalty: float = 0.00002
    reward_mtm_weight: float = 0.01

    # ── Parallelism & device (Track A2/A3, DECISIONS.md D11/D12) ───────────
    # torch_threads: torch's default is "all cores", which is pathological for
    # PPO's tiny matmuls (measured 15x slowdown from thread oversubscription on
    # a 20-core machine; compounding across parallel fold workers). Cap it.
    torch_threads: int = 4
    # Default device + vectorization width for train_sliding_walk_forward are
    # set from benchmarks.py measurements (see DECISIONS.md "A2/A3 sweep").
    # Measured (post-A1 env, 100k steps, this machine): cpu/4threads beats cuda
    # at every width — 5,036 vs 2,574 steps/s at n_envs=4; 10,377 vs 6,846 at 16.
    default_device: str = "cpu"
    # n_envs=4 x n_parallel_folds=4 fits the 20-core budget (4 x (4 envs + 1
    # main) = 20) and is the config validated by the 34-fold smoke sweep.
    # Single-fold runs should pass n_envs=16 (2x throughput) instead.
    default_n_envs: int = 4
    default_n_parallel_folds: int = 4

    # ── PPO regularisation (generalisation-first preset) ─────────────────────
    # The knobs that most directly control overfitting. Tune these between runs.
    # Rationale: the thin XAUUSD signal lets PPO memorise the train period if it
    # over-optimises, so this preset trains "softer" — more exploration, fewer
    # passes per rollout, a KL brake, a decaying LR, and explicit weight decay —
    # and leans on the best-checkpoint selector to stop before the late collapse.
    ppo_learning_rate: float = 6e-5
    ppo_lr_schedule: str = "linear"        # "linear" decays LR → 0 over training; "constant" holds it
    ppo_ent_coef: float = 0.03             # entropy bonus: higher = more exploration, less brittle policy
    ppo_n_epochs: int = 5                  # SGD passes per rollout: fewer = less per-batch memorisation
    ppo_clip_range: float = 0.1            # PPO trust region (kept tight)
    ppo_target_kl: Optional[float] = 0.025  # early-stop the epoch loop if policy moves too far (None disables)
    ppo_weight_decay: float = 1e-5         # L2 on the policy/value net (Adam optimizer_kwargs)
    ppo_net_arch: Tuple[int, ...] = (128, 64)  # smaller net regularises the thin signal

    # ── Track B3: JAX trainer geometry (DECISIONS.md D36) ─────────────────
    # Hyperparameters above map 1:1 onto jax_ppo.py.  Geometry mapping of the
    # SB3 reference run (n_steps=2048 x n_envs=4 = 8192 transitions/update,
    # batch 256, 32 minibatches/epoch x 5 epochs = 160 grad steps/update):
    # the JAX defaults trade rollout horizon for GPU batch width while keeping
    # 32 minibatches/epoch and 160 grad steps/update — 64 envs x 256 steps =
    # 16384 transitions/update (2x buffer), minibatch 512.  GAE horizon 256 vs
    # 2048 is negligible at gamma*lambda=0.9405 (tail weight ~1e-7).
    # SB3-parity preset (identical buffer + GAE horizon to the SB3 runs):
    # num_envs=4, num_steps=2048, num_minibatches=32 -> minibatch 256.
    jax_num_envs: int = 64
    jax_num_steps: int = 256
    jax_num_minibatches: int = 32

    # Baseline search space. Wider brackets reduce churn on noisy intraday gold.
    baseline_threshold_grid: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
    baseline_sl_idx_grid: Tuple[int, ...] = (1, 2)
    baseline_tp_idx_grid: Tuple[int, ...] = (2, 3)

    # Plotting.
    plot_bars: int = 500

    # ── Derived / read-only properties ──────────────────────────────────────

    @property
    def pandas_tf(self) -> str:
        """Canonical pandas resample rule for decision_timeframe.

        Accepts MT4/MT5 notation ("H1", "M5", "D1"), shorthand ("1h", "5m"),
        or a pandas rule directly ("1h", "5min").  Raises ValueError for
        unrecognised strings so misconfiguration fails loudly.
        """
        rule = _TF_TO_PANDAS.get(self.decision_timeframe)
        if rule is None:
            raise ValueError(
                f"Unknown decision_timeframe '{self.decision_timeframe}'. "
                f"Recognised values: {sorted(_TF_TO_PANDAS.keys())}"
            )
        return rule

    @property
    def pandas_execution_tf(self) -> str:
        """Canonical pandas rule for execution_timeframe."""
        rule = _TF_TO_PANDAS.get(self.execution_timeframe)
        if rule is None:
            raise ValueError(
                f"Unknown execution_timeframe '{self.execution_timeframe}'. "
                f"Recognised values: {sorted(_TF_TO_PANDAS.keys())}"
            )
        return rule

    @property
    def periods_per_year(self) -> int:
        """Trading bars per year at decision_timeframe — used for annualising
        Sharpe, Sortino, Calmar.  Derived from _BARS_PER_YEAR lookup so it
        automatically adjusts when you change decision_timeframe."""
        return _BARS_PER_YEAR.get(self.pandas_tf, 6_003)


CFG = ProjectConfig()
