# ANALYSIS.md — Codebase, GPU & Scalability Review

**Project:** XAUUSD M1 → RL-ready feature + bracket-trading PPO pipeline
**Date of analysis:** 2026-11-01
**Scope:** full source review (`*.py`, README, config), plus *measured* benchmarks on this machine while a live training run was in progress / after it stopped.

**Test machine:** 20 CPU cores, 62 GB RAM, NVIDIA RTX 5070 Ti Laptop (12 GB, CUDA 13.1), Python 3.12, `torch 2.15.0+cu132`, `stable-baselines3 2.9.2`, `gymnasium 1.4.0`, numpy 2.5.4, pandas 2.x. No `polars`/`pyarrow` installed.

---

## 1. Executive summary / verdict

| Dimension | Rating | One-line finding |
|---|---|---|
| Methodological rigour (backtest science) | **Strong** | Causal features + leakage test, embargoed splits, sliding walk-forward with stitched OOS equity, sealed holdout, deploy gate, consistency-based checkpoint selection. Above average for a hobby RL trading repo. |
| Code quality / maintainability | **Good** | Typed, documented, modular, config-driven, deliberate comments explaining *why*. No tests/CI, no `.gitignore`, unpinned deps. |
| RL implementation quality | **Good with gaps** | Correct SB3 idioms (VecNormalize travelling with checkpoints, SubprocVecEnv, Monitor). Gaps: no truncation bootstrap flag, redundant action dims, one wrong-checkpoint bug in the non-default path. |
| GPU usage today | **Nominal / mostly idle** | CUDA is selected (`device="auto"`), but the policy is a 128-64 MLP. Measured GPU util during training: **13–30 %**, 352 MiB. SB3 itself warns MLP-on-GPU is inefficient. |
| Throughput | **Modest** | Live run: **~1,650 env-steps/s** (4 procs). Bottleneck is *not* GPU — it is the per-step Python loop (SB3) + pandas-based observation. |
| Naive vs efficient? | **Neither extreme** | Not naive (documented 100–200× env optimisation already landed), but a 4× env win and a 2–4× vectorization win are still on the table with ~1 day of work. |
| Amenable to scaling (Python → Rust → JAX/JaxMARL)? | **Yes, unusually amenable** | Fixed in-memory dataset, 31-float obs, 36-value discrete action, tiny state, no physics. This is about the *best* case for a GPU-vectorized RL rewrite. |
| Recommended pick-up? | **Yes** | Methodology is worth keeping; performance headroom in Python is ~5–15×; a JAX route is credible if you want agent swarms. |

**Bottom line:** this is a carefully-engineered research pipeline whose *science* is ahead of its *systems*. It is not a throwaway prototype. The default full sweep (34 sliding folds × 3 M steps = **102 M env steps ≈ 17 h** at current speed) is the main practical drag, and it is fixable without abandoning Python.

---

## 2. What the project is

Pipeline (`run_pipeline.py` for baselines, `train_ppo.py` for RL, `final_holdout_eval.py` for the one-time reveal):

1. **Data ingress** — single MT4/MT5-style CSV, `data/XAUUSD_1m_EET.csv` (495 MB, **7,872,046 M1 rows**, 2004→2026). Broker "EET" timestamps are localized to `Europe/Helsinki`, shifted from bar-open to bar-close, deduped and sanity-filtered (`data_loader.py:28`).
2. **Resampling** — M1 → decision timeframe (`H1` by default) → **137,710 bars** (`data_loader.py:113`).
3. **Features** — 25 causal, ATR-normalised/stationary features + 6 position-state features = **31-dim observation** (`features.py:131`, `env_bracket.py:90`). Four collinear features were deliberately dropped (documented).
4. **Environment** — Gymnasium `BracketTradingEnv` (`env_bracket.py`): action = `MultiDiscrete([3, 3, 4])` = **36 combos** (direction / SL-ATR bucket / TP-R bucket), intrabar TP/SL resolved against M1 bars, fixed-fractional 0.5 % risk sizing, spread+slippage+commission modelled, pessimistic SL-first rule when both are touched inside one M1 candle.
5. **Training** — SB3 PPO, MLP `(128, 64)`, `n_steps=2048`, `n_epochs=5`, clip 0.1, target-KL 0.025, entropy 0.03, weight decay, linear LR decay, VecNormalize, `SubprocVecEnv(n_envs=4)`.
6. **Validation** — two schemes: block walk-forward (5 folds) and the default **sliding walk-forward** (34 folds: 5y train → 6m val → 6m test, slide 6m) with embargo, stitched OOS equity, and a **deployment gate** on out-of-sample folds.
7. **Checkpoint selection** — `_ConsistencyEvalCallback` scores `min(q_train, q_val)` with a drawdown penalty and a do-nothing (idle policy) guard (`train_ppo.py:142`).
8. **Diagnostics/artifacts** — `training_diagnostics.py`, `visualize.py` (Plotly), notebooks.

**Scale:** sliding folds have 31,102 train / 2,896 val / 2,968 test H1 bars each. At 3 M steps/fold that is **≈ 96 effective passes** over each 5-year training window.

---

## 3. GPU usage — what it actually does today

* The only GPU code path is `device="auto"` → `torch.cuda.is_available()` → `cuda` (`train_ppo.py:338`), handed to `PPO(...)`. **No custom kernels, no GPU environment, no vectorized env.** All GPU work is tiny MLP forward/backward.
* Observed during the live run: `nvidia-smi` **13–30 % util**, **352 MiB** of 12 GB (a few % of VRAM).
* SB3 emits an explicit warning every run (`on_policy_algorithm.py:150`):
  > *"You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy … the GPU utilization will be poor and the training might take longer than on CPU."*
* Measured end-to-end (see §4.3): the warning is directionally right *per call* (GPU inference is ~3× slower per call than CPU) but wrong *for this machine's default configuration*, because the default CPU path was 15× slower due to torch thread oversubscription (§4.3).

**Verdict:** the project "uses a GPU" in the checkbox sense only. ~85–90 % of the silicon is idle. Nothing in the design currently *needs* a GPU, and nothing currently *exploits* one.

---

## 4. Measured performance (the important part)

> Methodology: micro-benchmarks run with the project's own `build_env`/`train_ppo` code. First batch ran **while the live `python train_ppo.py` job was active** (pessimistic, marked "contended"); follow-ups ran on an idle machine. Live-run numbers come from the job's own `consistency_evals.csv` timestamps. During this analysis the live job stopped (fold 2 at 2.1 M/3 M steps, 21:23); no OOM evidence was readable — treat live numbers as approximate.

### 4.1 Data ingress

| Stage | Time | Notes |
|---|---|---|
| `pd.read_csv` (495 MB, 7.87 M rows) | **1.41 s** | single-threaded C parser, ~350 MB/s |
| `to_datetime(format="%Y.%m.%d %H:%M:%S")` | **0.98 s** | fast path with explicit format |
| `tz_localize("Europe/Helsinki")` | **0.20 s** | |
| **Full `load_mt_ohlcv_csv()`** (incl. coercion, filters, dedup, sort, OHLC sanity) | **3.5–6.6 s** | in-memory frame 378 MB |
| Resample M1→H1 + all features (`prepare_feature_frame`) | **0.28 s** | 137,710 bars — trivial |
| M1 window slice `.copy()` (0.8 split, 6.2 M rows) | **0.20–0.26 s** | done once per env construction |

**Assessment:** ingress is *not* the current bottleneck for one symbol at M1. It costs ~4–6 s **per entry point invocation** (`run_pipeline`, `train_ppo`, `training_diagnostics`, `final_holdout_eval`, `view_results` each re-parse the CSV from scratch) — a fixed ~30–40 s of duplication across a full workflow, and it does not scale to tick data or many symbols (§9).

### 4.2 Environment stepping

| Configuration | steps/s | µs/step |
|---|---|---|
| `BracketTradingEnv` alone, **as written** (idle) | **3,640** | 275 |
| … contended (live job running) | 2,521–3,344 | 300–400 |
| ↳ of which `_observation()` alone | — | **244 µs (89 % of the step)** |
| ↳ `decision_df.iloc[i]` row access alone | — | 30 µs |
| ↳ equivalent pre-computed **numpy** observation | — | **~1 µs** |
| `BracketTradingEnv` with observation/Close/ATR pre-extracted to numpy | **14,886** | **67 µs (4.1× faster)** |
| `SubprocVecEnv×4 + VecNormalize` (random actions), as written | **5,198** | — |
| `SubprocVecEnv×4 + VecNormalize`, numpy-patched | **10,545** | **2.0×** |

The M1 intrabar scan is already optimised (`searchsorted` + cached `float64` arrays, `env_bracket.py:83,233` — comments claim ~100–200× over a boolean-mask scan). The **remaining hot cost is building the observation from pandas every step**: `row[feature_cols].astype(float).to_numpy(...)` plus `np.nan_to_num` + `concatenate`, and `_current_row()` is called up to 4× per step (`env_bracket.py:119,148,266,292`).

### 4.3 End-to-end PPO throughput (4 envs, 100 k steps, MLP 128-64, batch 1024)

| Configuration | env-steps/s | wall |
|---|---|---|
| `device="cuda"` (idle) | **1,886** | 53 s |
| `device="cuda"` + numpy-patched env | **2,361** | 42 s |
| `device="cpu"` **default threads (20)** | **312–335** | 298–320 s |
| `device="cpu"` **`torch.set_num_threads(4)`** | **2,567–2,649** | 19 s |
| live project run (fold 1: 3 M steps; fold 2: 2.1 M steps) | **~1,600–1,750** | 28–29 min/fold |

Optimizer micro-benchmark (fwd+bwd+step, batch 1024): **CPU-1 thread 2.31 ms, CPU-4 threads 0.59 ms, CPU-20 threads 1.23 ms, CUDA 1.35–1.40 ms** — i.e. the GPU's optimizer advantage over a *correctly configured* CPU is small for this MLP; the huge CPU-vs-GPU gap in the table above is entirely thread oversubscription (20 torch threads hammering tiny matmuls in the main process while 4 env workers also burn cores).

**Where the per-step time goes** (50 k-step instrumented run, per *vector* step = 4 env steps in parallel):

| | CPU (4 threads) | CUDA |
|---|---|---|
| policy forward (`self.policy(obs)`) | 0.41 ms (27 %) | **0.92 ms (40 %)** — H2D/D2H + kernel launch per step |
| env worker compute + IPC (`step_wait`) | 0.63 ms (42 %) | 0.71 ms (31 %) |
| VecNormalize + callback + buffer + Python | 0.78 ms (31 %) | 0.67 ms (29 %) |
| gradient training (`PPO.train`) | 0.9 s total (5 %) | 1.1 s total (6 %) |
| **total** | **1.51 ms/vec-step → 2,649 steps/s** | 2.30 ms/vec-step → 1,739 steps/s |

**Key structural insight:** policy-forward + "other" are **fixed per vector step regardless of `n_envs`**. Going from 4 → 16 envs amortises 1.19 ms over 16 steps instead of 4, giving roughly 3–4× more env-steps/s at *zero* algorithmic change. The env itself (42 %) is also 4× improvable with numpy (§4.2).

### 4.4 Live-run economics

* Fold 1: 3 M steps ≈ **28–29 min** (20:32 → 21:01) ⇒ ≈ 1,700 steps/s, consistent with the benchmarks (and with a concurrent load).
* Default sweep: **34 folds × 3 M = 102 M steps ⇒ ≈ 17.2 h**, plus ~1 min/fold of post-training full-history rollout (110 k steps) and eval ⇒ **≈ 18 h per complete re-run**, single-fold-at-a-time, on 20 cores.

---

## 5. Is the implementation naive, or reasonably efficient?

**Verdict: reasonably efficient, methodologically strong, with identifiable headroom. Not naive.**

**Already good (evidence against "naive"):**
* M1 intrabar lookup pre-optimised to `searchsorted` + cached numpy (`env_bracket.py:79–85, 218–234`) with a documented 100–200× claim.
* `SubprocVecEnv` used, pickle-safe module-level factory, forkserver/spawn selection (`train_ppo.py:32, 362`).
* VecNormalize stats **snapshotted next to every saved checkpoint** (`train_ppo.py:265–268`) — a subtle correctness detail most projects miss.
* `_CaptureDoneWrapper` correctly placed to survive `DummyVecEnv` auto-reset (`train_ppo.py:536`).
* VecNormalize sync from train → eval envs before each checkpoint score (`train_ppo.py:121`).
* Reward normalised by risk budget, not cash PnL; dense MtM shaping at weight 0.01 with an explicit no-lookahead comment (`env_bracket.py:285–300`).
* PPO regularisation preset in config with rationale comments; per-fold timestep scaling to equalise passes; `target_kl` early stop.

**Naive / inefficient spots (ordered by impact):**

1. **Per-step pandas observation build** — 244 µs of a 275 µs step; a numpy pre-extraction (which the code already does for M1 high/low!) gives **4.1×** (`env_bracket.py:148–154`).
2. **`n_envs=4` leaves per-vector-step Python cost on the table** — 1.19 ms of fixed overhead per 4 steps (§4.3).
3. **Folds trained strictly sequentially** — the 34 folds are embarrassingly independent; 20 cores are used at 4-way concurrency at best.
4. **Torch thread defaults** — if anyone runs `device="cpu"` they get a silent **15× penalty** (`torch.get_num_threads()==20`); needs `torch.set_num_threads(4)` (or `OMP_NUM_THREADS`) pinned in code.
5. **Full 110 k-step deterministic train rollout per fold** for the in-sample/OOS chart (`train_ppo.py:514–530`) — ~45 s/fold at current speed.
6. **Memory churn** — `m1_df.copy()` inside the env (`env_bracket.py:64`) plus `_slice_m1_for_decision_window().copy()` (`train_ppo.py:43`) plus forkserver pickling of the args into each worker ⇒ several copies of an ~90–300 MB M1 slice per fold. Fine at 62 GB; naive at tick scale.
7. **Per-step `history.append(dict)`** in the env (`env_bracket.py:302`) — fine for 2048-step episodes, wasteful for the 110 k-step eval rollouts.

---

## 6. Scaling analysis — the question you actually care about

### 6.1 Tier 0 — configuration fixes (minutes of work, ~1.3–2× end-to-end)

* `torch.set_num_threads(4)` (or run `device="cpu"` with the same cap): measured **1,739 → 2,649 steps/s** vs. the 20-thread CPU default which collapsed to 312–335.
* Keep or drop CUDA: with capped threads, CPU (2,649) **beat** CUDA (1,739) in our A/B, because SB3's per-step `obs → tensor → forward → .cpu().numpy()` pays ~0.9 ms of transfer/launch on CUDA versus 0.41 ms on CPU. **GPU is not the lever under SB3.**

### 6.1b Tier 1 — Python-level, ~1 day, est. **5–10× end-to-end**

| Change | Est. gain | Risk |
|---|---|---|
| Pre-extract `feature_cols`/`Close`/`atr` to contiguous numpy in the env (measure: **4.1× env**) | env 4× → ~1.5–2× end-to-end | Very low (same semantics; add a parity test) |
| `n_envs` 4 → 16 (amortise fixed per-vec-step cost) | ~2.5–3.5× | Low; RAM = n_envs × (M1 slice + env copy); raise `batch_size` to 2048–4098 |
| Parallelise sliding folds (e.g. 5 folds × 4 envs on 20 cores) | ~4–5× wall-clock | Low; GPU would be shared (fine — GPU work is 6 %) |
| Cache M1+features to parquet/feather once; all entry points read the cache | ~5 s → <1 s per invocation, every run | Very low |
| Skip/shorten the per-fold 110 k-step chart rollout | ~1 min/fold (≈35 min/sweep) | None |
| Reduce `total_timesteps` if 96 passes/fold overfit (config already warns) | up to 3× cheaper | Methodological — check with a sensitivity run |

**Combined realistic estimate: 17 h → 1.5–3 h per full sweep**, no language change, no algorithm change.

### 6.2 Tier 2 — JAX / GPU agent-swarm (JaxMARL et al.), ~1–3 weeks, est. **10²–10⁴× env throughput**

**This environment is an unusually good fit for GPU vectorization.** Concretely:

* **State is tiny:** `(i, equity, position record)`. No physics, no contact dynamics, no raster obs. 31-float observation, 36-way discrete action.
* **Data is fixed and fits in VRAM:** the full 5-year M1 slice for one fold is ~1.8 M rows × 2 float64 arrays ≈ 30 MB; *all 7.87 M M1 rows* ≈ 126 MB — trivially resident on a 12 GB card.
* **The intrabar TP/SL loop vectorizes:** pad each H1 interval to its max 60 M1 bars (offsets via one `searchsorted` pass at setup), then first-touch detection becomes a `(batch, 60)` mask + `argmax`/`cumprod` — no Python loop, no per-step searchsorted. This is the single biggest algorithmic enabler and it preserves the pessimistic SL-first rule exactly.
* **`randomize_start` already gives you the swarm:** thousands of independent episode starts = thousands of `vmap`'d environments in one dispatch. On-policy PPO with a 128-64 MLP then runs with batch sizes (e.g. 1024 envs × 128 steps) that make GPU utilization near-100 % instead of 13–30 %.
* **The walk-forward structure is itself a swarm:** 34 folds × N seeds × hyperparameters = a natural population/batch dimension (this also directly addresses the single-seed weakness in §7).

**Framework fit:**
* **JaxMARL** is multi-agent-oriented (`vmap` over agents *and* envs). It fits if you reframe the problem as a swarm: e.g. one agent per symbol/timeframe/parameter-set, or population-based training over bracket variants. It will require adapting the single-agent PPO to its Sable/QMIX-style APIs or replacing the learner with a vanilla JAX PPO while keeping JaxMARL's env abstractions.
* **Better-matched single-agent tools:** PureJaxRL / `ppo_xla`-style JAX PPO, or a CleanRL-JAX single-file PPO — same `vmap` env, same GPU batching, no multi-agent overhead. Equinox (`jaxmarl`'s nn library) is a fine MLP home.
* **Also a natural for Evolution Strategies / CMA-ES:** ES is embarrassingly parallel, GPU-friendly, and would let you sweep seeds × folds × PPO hyperparameters far more cheaply than 34 sequential PPO runs.

**Risks/costs of the JAX route:** (a) you must re-derive `VecNormalize` (running mean/var) and reward normalisation in JAX; (b) exact semantic parity of the simulation (spread/slippage, SL-first, commission, equity chaining) needs a differential test against the pandas env — otherwise your OOS numbers are not comparable to the existing walk-forward records; (c) SB3's ecosystem (callbacks, checkpoint-consistency gate, diagnostics) must be rebuilt — that's the real cost, not the env; (d) debugging vectorized envs is harder.

**Expected outcome if done:** 10⁶–10⁷ env-steps/s is realistic for this env on a 12 GB laptop GPU (the whole 102 M-step sweep in minutes), which changes the science: you can afford seed sweeps, hyperparameter sweeps, and per-fold ensembles that are currently out of reach.

### 6.3 Tier 3 — Rust: where it helps and where it does not

* **Env-only rewrite (PyO3 module):** env step could go from ~67 µs (numpy-patched) to ~1–5 µs. But in the measured budget the env is only **42 %** of per-vector-step time and IPC/pickling stays ⇒ **~1.6× end-to-end inside SB3**. *Not worth it alone.*
* **Rust only pays if the driver loop also leaves Python** — i.e. a Rust rollout runner calling into a Python (or Rust) gradient step, or a fully Rust trainer (`tch-rs`/`candle`). That is a bigger rewrite than the JAX path with less community leverage for RL.
* **Where Rust (via `polars`/`arrow-rs`) *is* clearly justified: data ingress at scale** — multi-symbol/tick/years-of-tick workloads, columnar memory maps, and a shared cache readable from Python. This is the sensible Rust beachhead: `data_loader` → Polars/Arrow, keep RL in Python/JAX.

**Recommended sequencing:** Tier 0+1 (Python, days) → decide whether 2–3 h sweeps are enough → only then JAX (if you want swarms/seeds/ES) and Polars for ingress (if data grows). Rust for the env is the *last* thing to reach for.

---

## 7. Methodology review (matters for "should I build on this")

**Strengths**
* Causal feature engineering with an automated future-append leakage assertion (`leakage_checks.py`), 200-bar embargo at every split boundary sized to the longest EWM (EMA-200 rationale in `config.py:126`).
* Sliding walk-forward that honestly simulates "retrain every 6 months, trade the next 6 months", with contiguous test windows stitched into one compounded OOS curve (`train_ppo.py:1041`).
* Sealed holdout released exactly once (`final_holdout_eval.py`), byte-identical across both split schemes (`test_frac == 1 - train_frac - val_frac`).
* Deployment gate with breadth + floor + mean-Sharpe conditions, and a "gate only prevents bad deploys" design (`config.py:115`).
* Checkpoint selection anchored on the weaker of train/val legs with a do-nothing guard (an idle policy cannot win on low drawdown).
* Risk-normalised rewards, pessimistic intrabar fill rule, fixed-fractional sizing deliberately kept out of the agent's hands.

**Caveats / risks to be aware of before extending**
1. **Gate computed on the *test* windows** (sliding path, `train_ppo.py:1078`). The stitched OOS track record you headline is therefore *conditional on having passed a gate evaluated over those same windows* — a strategy-level selection effect. It does not leak into weights, but it inflates the expected honesty of the headline number. Consider gating on val and reporting test untouched, or reporting both.
2. **≈ 96 effective passes per fold** over the 5-year window (3 M steps / 31,102 bars). With a thin signal this is a lot of exposure; the regularisation preset mitigates but does not remove it. A timesteps sensitivity run (1 M vs 3 M) would be cheap insurance.
3. **Truncation is not bootstrapped.** `env_bracket.step` returns `truncated=True` at `max_episode_steps` but never sets `info["TimeLimit.truncated"]` (`env_bracket.py:313–317`), so SB3 treats the arbitrary 2048-step boundary as a true terminal state and zeroes the value target. For a continuous market sliced into random segments this biases the value function slightly at every episode boundary. Fix: set the flag (SB3's `DummyVecEnv` already stores `terminal_observation`).
4. **Bug in the non-default path:** `train_walk_forward()` evaluates the **final** weights under the **best-checkpoint's** VecNormalize (`train_ppo.py:837` returns `model`, `:853–857` prefers `best_model_vecnorm.pkl`) → the block-WF summary mixes two different normalisations. The default sliding path is correct (`_load_fold_model`, `train_ppo.py:1011`).
5. **Action-space redundancy:** bracket dims (`sl`, `tp`) only take effect at entry — while holding, all 12 bracket combinations for the current direction behave identically (`env_bracket.py:270–280`). Roughly ⅔ of the 36 actions are behaviourally identical in the "hold" state, which adds credit-assignment noise. Consider an explicit (direction × bracket) action only when flat / an explicit "hold" action, or re-bracket-on-adjust semantics.
6. **Single seed (42) per fold.** RL variance across seeds is usually larger than the effect sizes being measured; the repo's own `ensemble_analysis.ipynb` suggests awareness. The JAX/ES route (or just 3 seeds × fewer folds) fixes this cheaply.
7. **Fixed cost model** (spread 0.20, slippage 0.02) — fine, but no regime/volatility-dependent spread, so live costs will differ.

---

## 8. Data ingress — current state and how it scales

**Current:** `pd.read_csv` → string datetime parse → `tz_localize` → open→close shift → dedup/filters → sort (3.5–6.6 s, 378 MB resident). Everything is single-threaded, in-memory, and **re-done from the CSV in every entry point**. `requirements.txt` pins only lower bounds; no `pyarrow`/`polars`.

| Scenario | Verdict |
|---|---|
| One symbol, M1, 23 y (today) | Fine. Biggest win is a **one-time parquet/feather cache** (~0.3–0.8 s) consumed by all scripts; also lets you delete the repeated parse. |
| Many symbols / M1+M5+M15+H1 copies | Switch to **Polars/pyarrow** (multi-threaded read, ~3–5× faster) + columnar partitioned storage; compute features once and cache the H1 feature frame (137 k × 30 cols ≈ 40 MB) — *that* is the tensor the RL actually consumes. |
| Tick data / >10 GB | Arrow/Polars + memory-mapped columnar (or Rust `polars` front-end feeding Python); avoid pandas `read_csv` entirely; keep only the decision-frame in RAM and stream M1 windows per fold. |
| Parallel folds (Tier 1) | Each worker/fold needs its own M1 slice copy today (pickle + `m1_df.copy()`). Share via a read-only memory-mapped Arrow/npz file or `fork` + copy-on-write to avoid 4–8× duplication. |

**Scale assessment:** the *ingress code shape* is naive (reparse-per-script, deep copies, pandas-everything) but the *volume* is small enough that it costs seconds, not hours. Fix it early only if you plan multi-symbol/tick expansion — then Polars/Arrow is the right (and Rust-compatible) call.

---

## 9. Hygiene / tech debt list

* **No tests, no CI, no lint/type-check config** — for a project whose value is *trustworthy* OOS numbers, a parity test (env vs. numpy-env, split boundaries, leakage) is the highest-leverage missing artifact.
* **No `.gitignore`** — `data/` (495 MB), `.venv/`, `models/`, `outputs/` are untracked but not ignored; one `git add .` away from a disaster. Single commit in history.
* **Unpinned `requirements.txt`** (`stable-baselines3>=2.3` while 2.9.2 is installed) — reproducibility risk for a results-driven repo.
* Encoding artifact in `train_ppo.py:517` (`"â€¦"`), duplicate print in `train()`.
* Wrapper introspection `_CaptureDoneWrapper` digs `while hasattr(inner, "env")` (`train_ppo.py:554`) — fragile against SB3/gym upgrades (documented, but brittle).
* `evaluate.py` default `periods_per_year=252*24*12` disagrees with `CFG.periods_per_year` (6,003 for H1) — overridden in the paths that matter, but a footgun for new callers.

---

## 10. Recommendation

**Pick it up.** The methodology (walk-forward + embargo + sealed holdout + gated deploy + consistency checkpoints) is the hard-to-fake part and it is already in good shape; the performance work is mostly mechanical.

**Suggested plan:**

1. **Week 1 — make it fast and safe in Python (Tier 0+1):**
   `torch.set_num_threads(4)`; numpy observation path in the env (4.1×, with a parity test); `n_envs` 4→16 with `batch_size` 2048–4098; parquet cache for M1/features; parallel fold driver (5 folds × 4 envs); fix the `TimeLimit.truncated` bootstrap; fix the block-WF best/final mismatch; add `.gitignore` + pins + a couple of parity tests.
   *Expected: **17 h → ~1.5–3 h** per 34-fold sweep, and higher per-step throughput. That alone may be "scaled enough".*
2. **Week 2–3 — only if you want swarms:** prototype the JAX env (padded M1 window + `vmap` over ≥1,024 random starts) and validate against the pandas env on identical action sequences. If parity holds, swap in a JAX PPO (PureJaxRL/`ppo_xla`-style, or JaxMARL if you want the multi-agent/parameter-swarm framing). *Expected: full sweep in minutes, plus affordable seed/hyperparameter sweeps — the thing SB3 can never give you.*
3. **Ingress, when data grows:** Polars/Arrow (optionally behind a Rust core) with a shared columnar cache; do not invest in Rust for the environment itself unless Tier 2 is abandoned.

**One-line verdict:** *methodologically strong, moderately efficient, GPU-present-but-idle; clearly amenable to a 5–15× Python speedup now and to a 100×+ JAX/JaxMARL-style swarm later — a good candidate to pick up and move forward.*
