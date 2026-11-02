# DECISIONS.md — Track A implementation decision log

**Project:** XAUUSD RL pipeline — Track A "Python quick wins" (per `SCALING_PLAN.md`).
**Rule of this log:** every non-obvious decision made during autonomous implementation is recorded here with
its rationale, so it can be reviewed, challenged, or revisited. Newest entries at the bottom of each section.

---

## A0. Process decisions (made before touching code)

| # | Decision | Rationale |
|---|---|---|
| P1 | **Generate golden fixtures from the *current* environment before any code change.** | The refactor must be provably behaviour-preserving. Once `env_bracket.py` changes, the only way to prove parity is to have captured the old behaviour first. Fixtures = full step traces (obs/reward/termination/equity/trades) for scripted + random action sequences over deterministic synthetic data. |
| P2 | **Synthetic data for fixtures/tests, not the 500 MB CSV.** | Hermetic tests (no data dependency, fast CI), fully deterministic, and lets us construct edge cases (same-candle SL+TP, gap windows, truncation boundary) that real data may not exhibit at convenient indices. A real-data smoke test can be added later behind a marker. |
| P3 | **DECISIONS.md created before the first code edit; append-only during the session.** | Explicit user requirement ("record any decisions you make"). |
| P4 | **Track A includes the plan's "shared assets"** (LUT spec, parity harness, benchmark script, artifact conventions) since the plan assigns them to Track A and B/C depend on them. | `SCALING_PLAN.md` §"Shared assets" lists LUT spec owner = Track A. |

## A1. Environment refactor design (env_bracket.py)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Precompute observation inputs as contiguous numpy arrays at `__init__`:** feature matrix (float32), `Close` (float64), `atr` (float64). Build obs by indexing these arrays. | The measured bottleneck: `_observation()` = 244 µs of a 275 µs step, almost entirely `row[cols].astype(float).to_numpy()`. An equivalent numpy gather measures ~1 µs (ANALYSIS §4.2). Public API (`decision_df`, `i`, `position`, `equity_curve()`, `trade_log()`) stays unchanged. |
| D2 | **Sanitize NaN/inf in the feature matrix once at init** (`nan_to_num`), not per step. | Init already `dropna`s feature columns and `features.py` replaces ±inf with NaN before that, so the per-step `nan_to_num` is defensive-only. Same values guaranteed; removes per-step work. |
| D3 | **Vectorize the intrabar TP/SL scan with numpy window slices** (first-touch via masked argmax over `low`/`high` views of the window) instead of the per-bar Python loop. | The Python loop costs ~100 µs/step while in trade. The vectorized form is *exact* (same scan order, same SL-first tie rule) and cheap. **Not** a precomputed LUT — see D4. |
| D4 | **Do NOT implement the (bar × combo) first-touch LUT in the Python env.** The plan's LUT idea has a correctness subtlety: a position held across multiple decision bars re-checks **entry-bar** SL/TP prices against **later** windows, so the lookup key would be (entry_bar, window_bar, combo) = O(T²). The exact fix is a per-window monotone step-function (breakpoints sorted by low/high with running-min indices) — that is the design written into `LUT_SPEC.md` for the JAX/GPU port (where it is a gather). For the Python env, the vectorized window scan is simpler, exact, and already removes the loop overhead. | Correctness over theoretical speed; the LUT's real payoff is in the vectorized/GPU backends (Track B/C) where the same step-function formulation is cheap. `LUT_SPEC.md` documents the subtlety so B/C don't inherit the naive design. |
| D5 | **History recording: preallocated numpy arrays** (equity/realized/position/close/reward + start index & count) instead of `list[dict]`, with `equity_curve()` building the identical DataFrame on demand. | Preserves the exact output contract used by `_CaptureDoneWrapper`/`_rollout_on_split` while removing per-step dict allocation (~1–2 µs and GC pressure each step). Memory is bounded and cheap (≈ 5 MB even for 110 k-step eval episodes). |
| D6 | **Remove the redundant `m1_df.copy()` in `BracketTradingEnv.__init__`** (keep `decision_df` frame for API compat) and the redundant `.copy()` after `dropna`. | `_slice_m1_for_decision_window` already copies before construction; every env then copied the slice again (≈ 90–300 MB per env per fold). Only the extracted `float64` high/low arrays are needed for simulation; one copy of those (via `to_numpy`) is sufficient and callers only read. Documented in DECISIONS because it is a behaviour-relevant ownership change (env no longer mutates/copies caller frames — it never mutated them anyway). |
| D7 | **Keep `_current_row()`** as a thin compatibility shim (returns `decision_df.iloc[i]`) but remove all internal hot-path uses. | `baselines.py` reads `env.decision_df.iloc[env.i]` directly; third-party/eval code may use `_current_row`. No reason to break either. |
| D8 | **Add `info["TimeLimit.truncated"]`** on truncation in `step()` (A7.1 correctness fix, done in the same pass since it touches `info`). | SB3 bootstraps the value function at 2048-step episode boundaries only when this flag is present (`DummyVecEnv` supplies `terminal_observation`). Without it, GAE treats an arbitrary window cut as a true terminal — biasing the value function every episode. Additive change; does not affect env traces (parity fixtures unaffected). |

## A1b. Parity harness & fixtures

| # | Decision | Rationale |
|---|---|---|
| D9 | **Fixtures store per-step traces as `.npz`** (obs f32, reward f64, terminated/truncated bool, equity/realized f64, direction i8) plus trade-log columns as a `.csv`; scenario input data lives in a shared `tests/scenarios.py` module. | NPZ keeps numeric traces exact and compact; CSV keeps the trade log human-diffable. Input data generated in-code (seeded) so fixtures stay valid without shipping data files. |
| D10 | **Edge-case scenarios must be scripted**, not just random: all-flat; seeded random actions; tight-SL entry → immediate SL; same-candle SL+TP (asserts SL-first exit reason); flip long→short; manual close; truncation boundary (short `max_episode_steps`); entry near data end (termination). | These are exactly the semantics most likely to break under a vectorized intrabar rewrite (D3) and the history rewrite (D5). |

## A2–A3. Device, threads, vectorization width

| # | Decision | Rationale |
|---|---|---|
| D11 | **Do not guess defaults: measure after A1.** The benchmark suite will compare `cpu(torch threads=4)` vs `cuda` × `n_envs ∈ {4, 8, 16}` end-to-end, and the winner becomes the documented default in `config.py`/`train_ppo.py`. | The trade-off genuinely flips with the env speed: cuda wins on optimizer-heavy loads but pays ~0.5 ms per `predict`; cpu+4threads won at the old env speed (ANALYSIS §4.3). Post-A1 the loop is cheaper, so the balance may shift again. Evidence will be pasted into this file. |
| D12 | **Add `torch_threads` as an explicit config field** (default 4) applied in `train()`. | torch's default (all 20 cores on this machine) caused a measured 15× slowdown on tiny-matmul PPO updates via thread oversubscription (ANALYSIS §4.3). It must be pinned in code, not left to the environment. |

## A4. Fold parallelism

| # | Decision | Rationale |
|---|---|---|
| D13 | **Parallel folds via subprocess workers** (one OS process per fold running `train_ppo.py --fold-worker <k>`), orchestrated by the parent which then aggregates `fold_result.json` files per fold. | CUDA contexts cannot be shared across forked children reliably; subprocess isolation contains crashes and memory; fold artifacts already live in per-fold directories so the merge step is trivial. Threads are rejected (GIL + CUDA). |
| D14 | **Workers load data themselves (via the A5 cache) instead of the parent pickling M1 slices to children.** | Avoids serializing 90–300 MB per worker; the cache makes each load ~1 s. |
| D15 | **Fold concurrency default = 4, each with n_envs from A3, and the driver documents the core-budget rule** (`n_parallel_folds × n_envs ≤ physical cores − 2`). | 20 cores: 4 folds × 4 envs = 16 workers + 4 main processes fits. Final numbers come from measurement (D11). |

## A5. Data-ingress cache

| # | Decision | Rationale |
|---|---|---|
| D16 | **`data_cache.py` with parquet (pyarrow) + a config-hash key** (file size/mtime + parse params for M1; + feature params for the H1 frame), written to `data/cache/`. `rebuild`/`force` flags and a CLI. All five entry points route through it. | Plan A5. Parquet chosen over pickle (columnar, compressed, cross-tool). Cache key includes mtime+size so a changed CSV invalidates automatically. |
| D17 | **Add `pyarrow` and `pytest` to `requirements.txt`, and pin the versions actually installed** (A7.3). | Parquet needs pyarrow; pytest for the new test suite; pinning closes the reproducibility gap flagged in ANALYSIS §9. |

## A6. Eval overhead

| # | Decision | Rationale |
|---|---|---|
| D18 | **`post_eval_chart` flag on `train()`** (default on for single runs; the sliding driver passes `False` for all but the final fold). | Per-fold full-train rollouts (~110 k steps) cost ~35 min/sweep at old speeds; the chart is diagnostic value only. The production fold keeps its chart. If measurement shows the cost is negligible post-A1, the flag stays but the default may be revisited (recorded below if so). |

## A7. Hygiene

| # | Decision | Rationale |
|---|---|---|
| D19 | **Fix the block walk-forward bug** (`train_walk_forward` evaluates final weights under the best-checkpoint's VecNormalize) by routing through `_load_fold_model` like the sliding scheme. | ANALYSIS §7 finding #4 — a genuine metrics bug in the non-default path. |
| D20 | **`.gitignore`** for `data/`, `.venv/`, `models/`, `outputs/`, `__pycache__/`, `data/cache/`; fix the mojibake in `train_ppo.py` post-training prints. | ANALYSIS §9. |
| D21 | **Tests run with `python -m pytest tests/`** and must be green with no GPU and no real data (synthetic only). | Hermetic gate for everything else. |

---

## Measured evidence (to be appended as benchmarks land)

### A1 — env refactor (measured on this machine, synthetic scenario data)

| Configuration | Throughput | vs baseline |
|---|---|---|
| **Baseline** (pre-refactor, ANALYSIS §4.2) single-proc, random actions | 2,521–3,640 steps/s (275–400 µs) | 1× |
| **Post-A1** single-proc, random actions (incl. trades) | **37,854 steps/s (26 µs)** | **~15×** |
| **Post-A1** single-proc, flat (no intrabar) | **173,931 steps/s (6 µs)** | ~52× |
| **Post-A1** single-proc, long-hold (in-trade intrabar) | 48,030 steps/s (21 µs) | ~19× |
| `_observation()` alone | 244 µs → **4.3 µs** | 57× |
| `_current_row()` compat shim (unused in hot path) | ~12 µs | — |

**Parity proof:** all 10 golden fixtures (flat, random, tight-SL churn, same-candle SL+TP,
flip+manual-close, truncation boundary, data-end termination, missing-M1-window, 3-day gap,
randomised start) match bitwise on obs and within 1e-9 on floats — including exit reasons,
`start_i`, and the equity_curve() schema. Suite: `tests/test_env_parity.py` (28 tests total with
intrabar-oracle and split/embargo tests; 31 with cache tests).

### A5 — data cache

| | Time |
|---|---|
| Cold (parse CSV + write parquet, `load_all(force=True)`) | 6.57 s |
| Fresh parse only (`load_mt_ohlcv_csv`, page-cached) | 1.74 s |
| **Warm (`load_all()` from parquet)** | **0.24 s** (**27× vs cold, 7× vs fresh parse**) |

Cache: `data/cache/*.parquet`, 225 MB, 7,872,046 M1 rows + 137,710×25 feature frame.
Parity verified byte-identical to fresh computation (tests/test_data_cache.py). Keys respond to
source-file mtime/size and to parse/feature parameter changes.

### A2 — thread cap (D12) evidence, from the A4 smoke runs

| Parallel smoke run (34 sliding folds, 60k steps/fold, n_envs=2, 4 workers) | Per-fold throughput | Load avg |
|---|---|---|
| **Before** `torch.set_num_threads` (each PPO proc takes all 20 cores) | **~21 steps/s** (5,6320 steps in 47 s was fine, but aggregate collapsed) | **99** on 20 cores |
| **After** `CFG.torch_threads=4` | **~1,200 steps/s per fold** (56k steps in 47 s), 4 folds concurrent | **6.5** |

This is the D12 mechanism measured in situ: the cap is not a micro-optimisation but the difference
between a working and a thrashing multi-fold sweep. `torch.set_num_threads(CFG.torch_threads)` is now
applied inside `train()`.

### A2/A3 sweep — `benchmarks.py sweep` (100k steps, post-A1 env, quiet machine)

| device | torch threads | n_envs | PPO.learn end-to-end |
|---|---|---|---|
| cuda | 20 | 4 | 2,574 steps/s |
| cuda | 20 | 8 | 4,434 steps/s |
| cuda | 20 | 16 | 6,846 steps/s |
| **cpu** | **4** | 4 | **5,036 steps/s** |
| **cpu** | **4** | 8 | **7,777 steps/s** |
| **cpu** | **4** | 16 | **10,377 steps/s** |

**Decisions (D22):**
1. **`default_device = "cpu"`** with `torch_threads = 4` — CPU beats CUDA at every batch width for
   this MLP policy (SB3's own on-policy warning says as much). GPU remains selectable via
   `device="cuda"` / `"auto"`.
2. **Throughput scales with `n_envs`** (fixed per-vector-step Python costs amortise): 4→16 is ~2×.
   **`default_n_envs = 4` + `default_n_parallel_folds = 4`** for the sweep — that fills the 20-core
   budget exactly (4 × (4 env workers + 1 main)) and is the configuration validated end-to-end by
   the 34-fold smoke sweep. Single-fold runs should pass `n_envs=16` (2× per-job).
3. Core-budget rule (D15, restated): `n_parallel_folds × (n_envs + 1) ≤ physical cores`, and keep
   `torch_threads` at 4 regardless — the cap is what keeps multi-fold sweeps from thrashing.

### A4 — parallel fold driver smoke test

`train_sliding_walk_forward(n_parallel_folds=4)` over the real 34-fold sliding schedule with
60k steps/fold: subprocess workers (`--fold-worker spec.json`), per-fold `worker.log`,
`fold_result.json` + `test_equity.csv` + `test_trades.csv` artifacts, parent aggregation into
`sliding_walk_forward_summary.csv`, stitched `sliding_oos_equity.csv`, and the consistency gate.
Crashed workers are recorded as failed rows without killing the sweep (D13).

Process lessons recorded here because they cost debugging time (P):
* **P5: detached jobs must use `setsid … < /dev/null`** — `nohup` alone does not survive the shell
  tool's process-group kill on command timeout (nohup only ignores SIGHUP). A killed launcher took
  the whole sweep down once; `setsid` (new session) survives.
* **P6: generate golden fixtures before touching the code under test** (P1) — done; 10 fixtures
  covering truncation/termination/SL-first/empty-window/gap/random-start semantics, committed in
  `tests/golden/` and deliberately NOT gitignored (they are regression anchors, not build output).

---

## Track A status: COMPLETE

**Bottom line:** env 15× faster (bit-parity proven against pre-refactor golden traces), data ingress
27× warm, the thread cap turns multi-fold sweeps from thrashing (load 99) to healthy (load 6.5),
folds run in parallel subprocess workers end-to-end (34/34 in ~9 min at toy budget), CPU+4threads
is the evidence-based default over CUDA for this MLP policy, and the harness (gate, artifacts,
sealed test) is unchanged and green: 31 tests + 10 golden fixtures.

| Item | Status | Evidence |
|---|---|---|
| A1 numpy obs + vectorized intrabar + preallocated history | done | 15× env, golden parity ×10 |
| A2 device/threads | done | sweep table above; `CFG.torch_threads=4`, `default_device="cpu"` |
| A3 vectorization width | done | 2× from n_envs 4→16; sweep defaults 4 folds × 4 envs documented |
| A4 parallel fold driver | done | 34/34 folds, artifacts + gate + stitched OOS (94,893 rows) |
| A5 data cache | done | 27× warm load, byte-parity tested (tests/test_data_cache.py) |
| A6 post-eval chart flag | done | `post_eval_chart` param; skipped for non-final sliding folds |
| A7 correctness/hygiene | done | truncation bootstrap, block-WF checkpoint fix, pinned requirements, `.gitignore`, mojibake fix, README notes |
| Shared assets | done | `LUT_SPEC.md`, `ARTIFACTS.md`, `benchmarks.py`, `tests/golden/` fixtures |

Out of Track A scope (per `SCALING_PLAN.md`): full 3M-step production sweeps and the Stage 1 JAX port.
