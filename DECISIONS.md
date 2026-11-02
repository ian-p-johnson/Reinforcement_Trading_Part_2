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

---

# DECISIONS.md — Track B implementation decision log (JAX port, B0–B3)

**Scope of this pass:** `SCALING_PLAN.md` Track B steps **B0–B3 only** (B4 walk-forward driver
integration and B5 throughput/swarms explicitly excluded). Rule as per P3: every non-obvious
decision is recorded with its rationale. Numbers: P5–P8 process, D22–D43 design.

## B0. Environment smoke test — results & process

| # | Decision | Rationale |
|---|---|---|
| P5 | **Install `jax[cuda13]` into the existing `.venv` (deviation from the plan's "fresh venv" wording), with a `pip freeze` diff + full test-suite re-run as the safety net.** | The B2 parity harness needs `env_bracket` (gymnasium/pandas) and the JAX env **in the same process** — a second venv would force a second dependency set or cross-process test plumbing. JAX is purely additive here: the freeze diff confirmed the 60 pinned packages byte-identical (only jax + its nvidia wheel deps added) and all 28 existing tests stayed green. |
| P6 | **B0 gate PASSED on rung 1 of the fallback ladder** — `jax[cuda13]==0.12.0.1` wheels, no fallback needed. Evidence below. | Driver 590.48.01 / CUDA 13.1 (this machine, RTX 5070 Ti Laptop, sm_120) accepted the cuda13 plugin directly. The known Blackwell risks ("No visible GPU devices" on this driver with cuda12 plugin; sm_120 jit-cache bug in jax 0.10.x; "No reference output" on older jax) did **not** reproduce on jax 0.12.0.1. |
| P7 | **No performance-sensitive tests run without asking.** B0–B3 contain only correctness work (parity/unit tests, tiny smoke runs). Throughput benchmarks belong to B5 and will be scheduled with the user. GPU use is limited to smoke tests and small parity runs. | User requirement. |
| P8 | **New modules stay flat at repo root** (`jax_env.py`, `jax_ppo.py`), tests in `tests/` alongside the existing suite; `requirements.txt` gains a pinned Track-B block (D17 convention). | Matches the repo's existing single-level layout (`env_bracket.py`, `train_ppo.py`, …). No package restructuring in a porting track. |

### B0 evidence (this machine)

```
jax 0.12.0.1, devices: [CudaDevice(id=0)]   (RTX 5070 Ti Laptop, sm_120, driver 590.48.01, CUDA 13.1)
jit matmul f32 (matmul_precision=highest) vs numpy: max err 5.7e-06   (DEFAULT precision = TF32 → 1e-2; documented pitfall)
jit matmul f64 vs numpy: max err 0.0                                  (x64 path is bit-exact here)
vmap over 1000 elements: OK
lax.scan 1000 steps: OK
pip freeze: only jax/jaxlib/jax-cuda13-plugin/jax-cuda13-pjrt/ml_dtypes/opt_einsum/nvidia-* added; no pin touched
pytest tests/ after install: 28 passed
```

Note: `jax_default_matmul_precision` stays at "highest" for this project's numerical work (money
math / parity); TF32 is unacceptable for the sim even though the MLP would tolerate it.

## B1. Design decisions (plan B1 items 1–7, written before coding)

| # | Decision | Rationale |
|---|---|---|
| D22 | **API: gymnax-style functional env** — `reset(rng, params) -> (obs, state)`, `step(rng, state, action, params) -> (obs, state, reward, terminated, truncated, info)`, six values exactly as plan B1.1; auto-reset lives in a thin wrapper (gymnax `AutoResetWrapper` pattern) that preserves `final_observation`; the parity harness drives the **raw** step so traces line up 1:1 with `tests/scenarios.py::run_trace`. | Plan B1.1. Also fixes the truncation-bootstrap gap by construction (terminated vs truncated are separate outputs; no `info` flag archaeology). |
| D23 | **Precision split: environment simulation f64 (`jax_enable_x64`), policy/value network + optimizer + GAE f32.** Observation is assembled exactly like `env_bracket._observation`: sanitized f32 feature row + position features computed in f64 then cast to f32. VecNormalize-equivalent moments in f64 (SB3's numpy is f64 too); normalized obs/reward cast to f32 at the network boundary. | Plan B1.2 offers "full f64 parity first" vs f32 net; the parity harness is the arbiter and only the *sim* has a reference to match. Money math (equity chaining, risk units, prices) must be f64 — B0 measured f64 matmul bit-exact on this GPU. The net is new code with no SB3 bit-parity to preserve, and SB3 trains in f32; GAE in f32 matches SB3's rollout-buffer dtype exactly (D35's torch-reference test is the check). |
| D24 | **State = pytree of batched scalars per env**: `i` (int32 decision index), `steps` (int32), `equity`, `realized_pnl` (f64), position block (`direction` int32, `entry_price`, `sl`, `tp`, `units`, `risk_cash`, `tp_r` f64, `bars_in_trade` int32). No per-step Python objects, no history lists; episode-return accumulation is a `LogWrapper`-style wrapper (JaxMARL pattern), matching plan B1.3. | Plan B1.3. `sl_distance`/`sl_atr_mult`/`entry_time` are not needed for simulation (recomputable/deterministic from entry bar) but `entry_time` **is** needed for the trade log — it is derivable as `decision_index[state.i_at_entry]`; see D31: the exit step carries the reconstructed trade record in `info`. |
| D25 | **Dataset arrays live in `EnvParams` (static, replicated)**: features `[T,25]` f32, `close`/`atr` `[T]` f64, and — per `LUT_SPEC.md` §3.4's explicit recommendation — the **padded-window gather form** (not the step-function LUT): `m1_high_pad`/`m1_low_pad` `[T,W]` f64, `m1_ts_pad` `[T,W]` int64 ns, `win_len` `[T]` int32, built once on the host. | Plan B1.4 + LUT_SPEC §3.4 ("start here; only build the step-function LUT if profiling shows the scan matters — it should not"). The gather is O(W) arithmetic per step, exact, and trivially `vmap`-able; the step-function LUT (§3.1–3.2) remains the documented fallback if B5 profiling ever says otherwise. Padding is ±inf so it can never register a touch (D30). |
| D26 | **Normalisation = exact `RunningMeanStd` port** (moment-combination algorithm, count starts at 1e-4, f64) for obs **and** discounted-return reward, per-step updates over the env batch (SB3 updates `ret_rms` per step too), `clip_obs=10`, `clip_rew=10`, reward norm **without** mean subtraction (`clip(rew / sqrt(var + 1e-8))`), frozen (`training=False`, `norm_reward=False`) for evaluation. Stats are part of the training state and snapshot **with** every saved checkpoint. | Plan B1.5 + `ARTIFACTS.md` invariant 1. Formulae read verbatim from SB3 2.9.2 `vec_normalize.py` / `running_mean_std.py`. The `_SyncVecNormCallback` semantics reduce to "eval reads the training state's snapshot at eval time" — trivially satisfied since eval is a separate jitted function (D40). |
| D27 | **Action semantics frozen**: `MultiDiscrete([3,3,4])` mapped exactly as today (0=flat/1=long/2=short; bracket dims consumed **only at entry**, ignored while in position — `env_bracket.step` passes them to `_open_position` only). Deterministic eval = per-dim argmax (SB3 `MultiCategorical.mode`). Logits layout `[3,3,4]` concatenated = 10. | Plan B1.6 explicitly forbids "improving" the action space in the same change. |
| D28 | **Episode regime preserved**: `randomize_start` with the identical headroom rule (`headroom = max(max_episode_steps+2, 2)`, `max_start = max(T - headroom, 1)`, `i = randint(0, max_start)`), `max_episode_steps=2048` truncation **with value bootstrap**, `terminated` at `i >= T-2` (data end) **without** bootstrap. | Plan B1.7. Bootstrap split verified against SB3 2.9.2 `collect_rollouts` semantics (`TimeLimit.truncated` → terminal-value bootstrap; hard termination → zero). |

## B2. Environment implementation decisions

| # | Decision | Rationale |
|---|---|---|
| D29 | **Window boundaries precomputed on the host** with the same `searchsorted(side="right")` half-open `(decision[i], decision[i+1]]` rule as `env_bracket._simulate_m1_until_next_decision`, baked into the padded arrays (row `i` of the pads *is* window `i`). The JAX step does no searching — just gather row `state.i`. | Exactness by construction: the builder can be diffed against the reference scan on the host before anything reaches the device. Precompute-once also removes the only non-trivial host-side logic from the step. |
| D30 | **Padding + tie rule**: `m1_high_pad` padded with `-inf`, `m1_low_pad` with `+inf` (never touches any level); first-touch = `argmax` of the boolean masks with `any()` guard; SL wins same-bar ties via `sl_rel <= tp_rel`; empty windows (`win_len=0`) hold the position. `W = max(win_len)` over the dataset, assert `W ≤ 1024` (sanity bound; H1/M1 physics caps it at 60 unless decision rows are dropped mid-series — measured W recorded in the build log). | Same semantics as D3's vectorized scan (itself golden-tested against the original per-bar loop). ±inf padding makes the mask unnecessary for correctness; kept as an `any()` guard only. The 1024 assert is a loud failure if a future dataset violates the H1→M1 assumption instead of a silent truncation. |
| D31 | **Trade-log reconstruction via fixed-shape `info` pytree**: on the step that closes a position, `info` carries the full record the pandas env would append (entry/exit times as int64 ns, entry/exit prices, sl, tp, sl_atr_mult, tp_r, units, pnl, r_mult, bars_in_trade, exit_reason id). A Python reconstructor turns per-step traces into the same trade DataFrame. | `lax` cannot return ragged trade lists; the fixed-size per-step record is the standard scan-friendly encoding and keeps the env pure. The parity harness then compares trade tables column-by-column against `trade_log()`. Exit times come from `m1_ts_pad` (D25). |
| D32 | **FP/branch-tie policy**: (a) the step's money math is f64 with identical operation order to `env_bracket` (so numpy and XLA round identically except for possible FMA contraction); (b) **edge-case scenarios that test exact ties** (`same_candle_sl_tp`, boundary `≤`/`≥` inclusivity) are constructed with exactly-representable prices (e.g. spread/2+slippage = 0.125, ATR = 10.0) so no rounding occurs and fusion cannot flip a comparison; (c) standard scenarios assert **exact** equality on bool/int fields and `atol=1e-9` on f64 traces (obs f32 `atol=1e-6`), with any observed exactness recorded. | XLA may legally fuse `a*b+c` into FMA (1-ulp differences vs numpy's sequential rounding); at genuine floating ties that could flip `low <= sl`. Exact-representable construction removes the hazard where the test's *point* is the tie; tolerance policy everywhere else is documented rather than pretending cross-toolchain bit-identity. |
| D33 | **Parity suite runs on both CPU and GPU backends** (parametrized), CI default CPU (`JAX_PLATFORMS=cpu` friendly). | Semantics must not depend on the backend; the GPU run is the cheap guard against platform-specific compilation surprises (B0's f32 matmul finding shows backends do differ numerically). |

## B3. Trainer decisions & deliberate deviations

| # | Decision | Rationale |
|---|---|---|
| D34 | **Structure = PureJaxRL `ppo.py` skeleton**: `vmap(reset/step)` over envs × `lax.scan(NUM_STEPS)` rollout → reverse-`scan` GAE → nested `scan` epochs/minibatches; the whole `train(rng, params, data)` wrapped in one `jax.jit`. Entropy/logging via light `jax.debug`/host callbacks — no chatty per-update prints inside `jit`. | Plan B3 prescribes exactly this. One `jit` closure per config (B5's JIT-hygiene note) — varying values (LR, fold data, rng) are **arguments**, never closure captures. |
| D35 | **Network matches SB3 2.9.2 `MlpPolicy` exactly**: separate (non-shared) pi and vf MLPs `[128,64]` Tanh (a plain `net_arch` list is separate nets in SB3 — verified in `torch_layers.py`), logits head `Linear(64, 10)` for `MultiDiscrete([3,3,4])`, value head `Linear(64,1)`, orthogonal init with SB3's gains (√2 hidden, 0.01 pi-head, 1.0 vf-head), zero biases. | Hyperparameter mapping must be 1:1 (plan B3); a torch-vs-JAX single-update unit test (D39) is only meaningful if the architecture and init are identical. |
| D36 | **Geometry mapping (documented deviation, the plan delegates the choice)**: `n_steps=2048` × SB3 `n_envs=4` → reference buffer 8192. JAX defaults **NUM_ENVS=64 × NUM_STEPS=256** = 16,384 transitions/update (2× SB3 buffer for GPU batch width), minibatch 512 → 32 minibatches/epoch → **160 grad steps/update, identical to SB3's** (256-batch × 32 × 5 epochs). `NUM_UPDATES = ceil(total_timesteps / (NUM_ENVS·NUM_STEPS))`, so the 3M-step budget semantics carry over. An **SB3-parity preset** (4×2048, minibatch 256 — buffer and horizon exactly SB3's) is used by the parity tests. All values are `config.py` fields (`jax_num_envs`, `jax_num_steps`, `jax_num_minibatches`). | Deviation from `n_steps=2048` exactly: rollout horizon 256 vs 2048 changes GAE tail weight `(γλ)^k` from 1.5e-7 to ~0 — negligible (and SB3's own default is 2048 with far fewer envs). Keeping grad-steps/update and minibatch-count at 160/32 preserves the update economics; the 2× buffer is the price of sane GPU occupancy. Recorded per plan B3's "deliberate deviations to resolve during design (document each)". |
| D37 | **`target_kl` early stop = `lax.while_loop` option (ii)**, at **SB3's granularity**: the loop over minibatches carries a `continue_training` flag; once `approx_kl > 1.5 · target_kl` (SB3's threshold) all further minibatches/epochs are skipped. `approx_kl = mean((exp(log_ratio) − 1) − log_ratio)` from the **pre-update** log-probs of the current minibatch, exactly as SB3 computes it after `optimizer.step()`. | Plan B3 deviation list, "choose the simplest that preserves the intent (bounded policy movement)". Option (i) (fixed epochs + post-hoc logging) does **not** bound movement; option (iii) loses the feedback loop. A `while_loop` with fixed max iterations is simple, static-shaped, and matches SB3's stop semantics almost exactly — the only residual difference is that SB3 checks inside the optimizer loop with the same formula (no deviation in practice: check happens at identical points). |
| D38 | **Optimizer equivalence (exact-by-construction)**: `optax.adam(learning_rate, eps=1e-8)` (SB3 torch-Adam defaults: β=(0.9, 0.999), eps=1e-8 — verified) with **coupled** L2 via `optax.add_decayed_weights(1e-5)` **before** adam (matches `torch.optim.Adam(weight_decay=…)` semantics, *not* decoupled AdamW — a real SB3-vs-AdamW difference), global grad-norm clip 0.5 (SB3 default), LR = linear 6e-5 → 0 over total updates (port of `_linear_schedule`). Single-update parity vs a torch recomputation is unit-tested (D39). | Plan B3: "unit-test a single update against a reference rather than trusting signs". The AdamW/Adam-L2 trap is exactly the kind of silent mismatch that test exists for. |
| D39 | **Unit tests for the math**: (a) GAE vs a small numpy reference on random data (same f32 semantics as SB3's `compute_returns_and_advantage`, incl. truncation bootstrap); (b) one full PPO update (loss + grads + params) vs a torch recompute on the same batch; (c) `RunningMeanStd` port vs SB3's class on random streams. | Plan B3. These pin the three places where silent semantic drift is most likely. |
| D40 | **Checkpoint selection = separate jitted eval functions driven from Python at `eval_freq`** (the plan's stated implementation choice): deterministic episodes on the train-tail slice and val slice, `q_leg = r_leg − dd_penalty·max_dd_pct_leg`, `score = min(q_train, q_val)`, eligibility = both legs profitable and ≥ `min_trades` (5) — a verbatim port of `_ConsistencyEvalCallback`; best checkpoint saved with its norm-stats snapshot; `consistency_evals.csv` written with the identical column schema. | Plan B3. Keeps `jit` shapes stable (no `lax.cond`/checkpoint writes inside the train `jit`); the CSV stays consumable by `training_diagnostics.py` (ARTIFACTS.md schema). |
| D41 | **Metric-parity test interpretation (documented)**: "eval-rollout metrics match SB3's" is validated as a **same-policy chain test** — a fixed deterministic action sequence (from a real Track-A fold model where available, else scripted) is replayed through the pandas env and the JAX env and both outputs are run through `evaluate.full_report`; results must match within D32's tolerances. Cross-trainer model comparison (JAX-trained vs SB3-trained policy returns) is **not** a parity test — different trainers legitimately produce different policies. | The plan's literal wording ("match SB3's numbers on the same fold") is only well-defined given a shared policy; this is the interpretation that validates "the whole chain" (env + reward + equity + metrics) without asserting a falsehood about training dynamics. |
| D42 | **Artifacts: same directory layout and `run_info.json` keys as `ARTIFACTS.md`** (new keys only, never renamed): params + norm stats as `best_model/best_model.npz` + `best_model/best_model_norm.npz` alongside the familiar paths, `run_info.json` gains `"framework": "jax"` so downstream scripts branch on the loader only (the plan's B4 note). `eval_logs/consistency_evals.csv` per D40. | `ARTIFACTS.md` invariants 1–3. B4 (out of scope) will wire the loader branches into `final_holdout_eval.py`/`training_diagnostics.py`. |
| D43 | **Reward/probability hygiene**: action log-prob for `MultiDiscrete` = **sum over the 3 dims** of per-dim categorical log-probs; entropy likewise summed (matches SB3 `MultiCategoricalDistribution`). Sampling uses the rng passed to `step` (gymnax convention) so `vmap` over envs gets independent draws. | Easy to get wrong (mean vs sum over dims) and silently scales the entropy bonus and KL by ~3×; verified against SB3 `distributions.py` during implementation and pinned by D39's update test. |

**Status:** B0 complete (evidence above), B1 frozen — implementation follows this document; any
change of mind gets a new numbered row, never a silent edit.

## B2/B3 — mid-implementation decisions (append-only, discovered while coding)

| # | Decision | Rationale |
|---|---|---|
| D44 | **Truncation bootstrap replicates SB3's exact chain, including `env_bracket`'s zero-obs-on-done quirk.** The env returns a zero observation when `terminated or truncated` (parity with `env_bracket.step`), so the bootstrap value is `V(normalise(zeros))` — which is precisely what the current SB3 pipeline computes: `on_policy_algorithm.py:236-245` adds `γ·V(terminal_observation)` to the buffered reward at `TimeLimit.truncated` steps, and `vec_normalize.py:203-204` normalises that terminal obs first. The gymnax `final_observation` returned by `step_autoreset` is therefore the zeros obs too. | Plan B1.1 promises "proper truncation bootstrap"; the *form* (fold the bootstrap into the reward at truncated steps, GAE treats dones as episode boundaries) is SB3's actual behaviour. The zero-obs quirk is preserved rather than fixed — it is what every existing SB3 number in this repo was produced with; changing it is a deliberate experiment, not a port. |
| D45 | **Orthogonal initialisation runs a host-side numpy QR, not `jax.nn.initializers.orthogonal`** (`jax_ppo._orthogonal`). | Blackwell/sm_120 finding: cuSolver's `cusolverDnDgeqrf` raised `INTERNAL` errors mid-suite on this machine (driver 590.48.01, jax 0.12.0.1) and poisoned the context so *later* unrelated kernels failed. The only QR in the trainer is the orthogonal init; numpy's QR is deterministic and equivalent. Same class of quirk as the B0 scan warned about — recorded so Track C (cuTile) knows cuSolver is flaky here. |
| D46 | **Trade records use a fixed 2-slot `info` pytree per step** (slot 0 = decision-close record, slot 1 = intrabar record). | A flip-close followed by an intrabar SL/TP exit closes **two** positions in one step — `env_bracket` appends two trade rows. A single record per step silently loses one (caught while designing the parity reconstructor; scenario `flip_and_close`-class paths exercise it). |
| D47 | **Timestamps travel as `index.asi8` integers with unit-aware conversion** (`jax_env.stamps_from_ts`). | pandas 3.x DatetimeIndexes are commonly `datetime64[us]` — `asi8` is *microseconds*, and a hard-coded ns assumption misread every trade time by 1000× (caught by the golden trade-log comparison). Simulation itself is unaffected (all comparisons stay in one unit); only the reconstruction boundary needs the index's unit. |
| D48 | **B3 "metric parity" test = same deterministic policy through both engines into `evaluate.full_report`** (`tests/test_jax_env_parity.py::test_metric_parity_full_report`, using a real Track-A fold model's captured action sequence over the fold's test window). | D41's interpretation. The alternative reading (JAX-trained policy metrics must equal SB3-trained policy metrics) is not a well-defined test — different trainers legitimately converge to different policies. |
| D49 | **Parity gate reuses `make_golden.compare`'s tolerance policy** (f32 obs and all ints exact, f64 `atol=1e-9`) rather than the D32 draft's looser f32 tolerance. | Observed diffs are bit-exact (0.0) everywhere measured, so the stricter existing policy holds; reusing the Track-A comparator keeps one parity standard across engines. D32's exactly-representable tie construction stands (verified bit-identical at exact `≤`/`≥` boundaries in `test_exact_tie_boundary`). |
| D50 | **`optax==0.2.8` added to requirements (pinned)** alongside the jax pins; installed into `.venv` with the same freeze-diff check as P5. | Needed for the D38 optimizer chain; pinning per D17. |

## Track B0–B3 status: COMPLETE (B4/B5 out of scope for this pass)

| Item | Status | Evidence |
|---|---|---|
| B0 JAX smoke (this machine) | done, rung 1 of fallback ladder | `CudaDevice(0)` sm_120, f64 matmul bit-exact, vmap/scan OK, all 31 pre-existing tests still green after install |
| B1 design decisions | done | D22–D43 above, written before coding |
| B2 env port + LUT construction | done | `jax_env.py`; padded-window gather (LUT_SPEC §3.4) + host window builder validated against the reference scan (10 scenarios × 200 samples) |
| B2 parity harness (hard gate) | done | 41 tests: 10 golden fixtures + 10 live-pandas deep compares + 5 random seeds + 3 exact-tie boundaries (bit-identical) + window validation + vmap-vs-loop + full sliding-fold rollout vs `_rollout_on_split` (real model) + `full_report` metric parity. **Observed trace diff: 0.0 (bit-exact)** on every scenario, both `JAX_PLATFORMS=cpu` and `cuda` |
| B3 trainer port | done | `jax_ppo.py`: PureJaxRL structure, SB3-verified network/loss/normalisation mappings, D37 KL stop, D40 consistency checkpointing with norm snapshots |
| B3 unit tests (D39) | done | GAE vs SB3 transcription (1e-5), single update vs torch (grads 1e-2 scale-relative; optimizer chain vs independent numpy Adam(wd) transcription 1e-5), RunningMeanStd vs SB3 class (1e-12), VecNormalize stream vs SB3 (1e-6), train smoke with full artifact contract |
| Full suite | green | 77 passed (31 Track-A + 41 B2 + 5 B3), combined run incl. the cuSolver-survivability fix D45 |
| Deliberate deviations logged | done | D36 geometry, D37 KL granularity/formula, D23 precision split, D44 bootstrap quirk, D48 metric-parity interpretation — plus `config.py` comments |

**Not done (out of scope, per plan):** B4 walk-forward driver integration, B5 throughput
validation & swarms. The trainer entry point is `jax_ppo.train_jax(...)` (accepts the
same `datasets` tuple as `train_ppo.train`) and writes the ARTIFACTS layout with
`"framework": "jax"` for the future loader branches.

---

# DECISIONS.md — Track B4/B5 decision log (driver integration & swarms)

## B4. Walk-forward driver integration

| # | Decision | Rationale |
|---|---|---|
| D51 | **Engine routing replaces only the inner `train()` call.** `train_sliding_walk_forward(engine="sb3"|"jax")` → `_run_sliding_fold(engine=...)` branches at the training call; the post-train tail (checkpoint load, val/test rollouts, `fold_result.json`/`test_equity.csv`/`test_trades.csv`, summary, stitching, gate, promotion/NO_DEPLOY) is shared code. Loader branches: `jax_ppo.JaxModel` handle returned by the framework-aware `_load_fold_model`, recognised by `_rollout_on_split`/`_run_split` (training_diagnostics) and by `final_holdout_eval._run_rl_holdout_jax`; `_promote_fold_to_production` is engine-aware for artifact names (npz pair vs zip/pkl). CLI: `python train_ppo.py --engine jax`. | Plan B4: "keep everything unchanged in Python… replace only the inner `train()` call… add a loader branch for the JAX checkpoint rather than rewriting those scripts." One shared tail guarantees the gate/stitching semantics cannot drift between engines. |
| D52 | **Parallel JAX fold workers set `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `XLA_PYTHON_CLIENT_ALLOCATOR=platform`** before any jax import; `engine` travels in `fold_spec.json`. Sequential mode verified first, then the parallel driver (2 workers) — both green in `tests/test_jax_driver.py`. | Plan B4 leaves this open ("decide during B4"): N jitted trainers each preallocating 75% VRAM cannot coexist; allocate-on-demand lets workers share the 12 GB card (per-worker working set is small: fold pads ~30–100 MB + nets). |
| D53 | **B5-L1 seed sweeps run sequential in-process** (`swarm.run_seed_sweep`), not concurrent — deviation from the plan's "N concurrent single-seed trains" wording. | On a single GPU, concurrent trainers time-share the same device anyway; sequential gives deterministic wall-clock attribution, trivial VRAM use, and no XLA allocator contention. Concurrency stays available via the fold-worker pattern if a multi-GPU box appears. |
| D54 | **B5-L3 is wrapper-style population hyperparameter search** (evosax 0.3.2 `SimpleGA`/`Open_ES`/`PGPE`/`CMA_ES` over (lr, ent_coef, clip_range), fitness = short-train + deterministic val `q`), **not weight-space ES/PBT**. | The plan offers "population/PBT over PPO hyperparameters **or policy weights**". Hyperparameter-level PBT is the robust, immediately useful half and reuses evosax ask/tell cleanly; weight-space evolution of a PPO policy needs a different fitness protocol (lengthy evals, nonstationarity) — recorded as open (B5 continuation). |
| D55 | **evosax multi-device sharding is not exercised** (single-GPU box). | Plan calls it "a one-liner"; no hardware to validate on — recorded rather than claimed. |
| D56 | **B5-L2 batched window scoring uses padded batches + per-window length masks** (`swarm.make_window_batch`): windows differ in both T and window-width W; both axes are padded (high −inf / low +inf / win_len 0 / zeros elsewhere), inactive steps freeze the state and zero the outputs. Verified exactly equal to the per-window loop. | "All test windows scored in one jitted call" needs uniform shapes; the inert-padding + mask design keeps semantics identical to the single-window eval (asserted in `test_batched_gate_eval_matches_loop`). |

## B5. Throughput validation — implementation status

| Item | Status | Evidence |
|---|---|---|
| Benchmark harness | implemented, **timings NOT yet run** | `jax_benchmarks.py`: `env` (vmap step @ 1k/4k/16k/64k envs), `train` (end-to-end steps/s @ CFG geometry), `comp` (compile time per shape), `xfer` (trajectory MB + pull cost), `vram` (residency/peak). JIT hygiene per plan: `clear_caches()` between experiments, static shapes, values as args (D34). |
| Harness wiring test | done (quick sizes only) | `tests/test_jax_swarm.py::test_benchmark_harness_wires` — asserts every group records output; **numbers at `--quick` sizes are never quotable**. |
| Swarm L1 (seeds) | done | `swarm.run_seed_sweep` + smoke test (2 seeds, synthetic) |
| Swarm L2 (batched gate evals) | done | `swarm.batched_gate_eval`, exact-match test vs per-window loop |
| Swarm L3 (population hp sweeps) | done (experimental) | `swarm.run_evosax_sweep`, evosax 0.3.2 API verified (`init/ask/tell`); smoke test pop 3 × 2 gens |
| **Recorded throughput numbers** | **PENDING — paused for a quiet machine** | To run: `python jax_benchmarks.py all --data real` (+ the B5 full-sweep wall-clock with `python train_ppo.py --engine jax` at production budget). Per the user's instruction, timing-sensitive runs are scheduled, not improvised. |

**B4 evidence:** `tests/test_jax_driver.py` — real-data toy-budget sliding runs into isolated
out_dirs: sequential JAX driver, parallel JAX driver (2 workers), and SB3↔JAX **schema parity**
(summary columns, fold_result keys, CSV columns identical; metrics legitimately differ across
trainers — D48). Loader branches exercised via the shared `_rollout_on_split` tail.

| D57 | **Eval-path `PPO.load(...)` calls go to CPU** (`_load_fold_model`, `final_holdout_eval._run_rl_holdout`, `training_diagnostics`), training device selection is unchanged. | SB3's `PPO.load` defaults to auto-device (CUDA if present). In shared-process runs JAX holds VRAM and torch's cuBLAS fails to initialise (`CUBLAS_STATUS_NOT_INITIALIZED`, seen in the combined suite). Single-env deterministic evals gain nothing from the GPU anyway (ANALYSIS §4.3: per-step transfer cost dominates). |
