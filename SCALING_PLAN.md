# SCALING_PLAN.md — Detailed execution plan (three tracks)

**Companion to:** `ANALYSIS.md` (measured baseline & code review) and `SCALING_GPU.md` (framework scan).
**Purpose:** concrete, task-level plans for the two-pronged attack plus its foundation:
**A) Python quick wins · B) JAX port · C) cutile/cuda-oxide spike.** No code in this document — design decisions,
task breakdowns, verification, gates, effort estimates, risks.

**Sequencing at a glance:** A starts immediately (days, de-risks everything). B and C both start after their
respective smoke tests and run as parallel prongs; A's outputs (LUT spec, parity harness, benchmarks) are shared
assets that B and C both consume. Nothing in C is on the critical path for B.

```
Week:   1            2            3            4            5            6
A:   [==== quick wins + parity harness ====]
B0:       [JAX smoke]
B:             [== env design+parity ==][== trainer+driver ==][swarms/bench]
C0:       [toolchain spike]
C:             [===== env kernel spike =====][seam decision / go-no-go]
```

---

## Shared assets (built once, used by all three tracks)

1. **First-touch LUT specification** (see `SCALING_GPU.md` §1): for every decision bar `i` and each of the
   24 `(direction, sl_mult, tp_r)` combos, the index (or offset) of the first M1 bar in the following window that
   touches SL or TP, computed with entry = close ± (spread/2 + slippage). Produced once on the host from the
   existing pandas data. *Owner: Track A (it pays for itself in the numpy env), consumed by B and C.*
   Deliverable: written spec (inputs, shapes, dtype, edge cases: window shorter than 60 (gap/weekend), both
   touched in same M1 → SL-first rule, no-touch sentinel) + a reference implementation validated against the
   current `_simulate_m1_until_next_decision` scan.
2. **Golden-vector parity harness**: a fixed set of (seed, start index, action sequence) episodes replayed through
   *any* environment implementation, emitting the full per-step trace (obs, reward, terminated/truncated, equity,
   trade log, exit reasons) as an artifact. Tolerance policy defined once (f64 → near-bitwise on equity after
   N steps; document any permitted drift). *Every track's "definition of done" includes passing this.*
3. **Benchmark suite**: the micro-benchmarks from `ANALYSIS.md` §4 (single-proc env, vec-env, end-to-end
   `PPO.learn` variants, data-load timings) kept as a runnable script so before/after numbers are comparable
   across tracks.
4. **Artifact conventions**: `run_info.json`, per-fold summary CSV schemas, `best_model` layout, gate columns —
   frozen now so every track can drop into the existing harness (`train_sliding_walk_forward` gate,
   `final_holdout_eval.py`, `training_diagnostics.py`) without renegotiating formats.

---

# Track A — Python quick wins (3–5 focused days; sweep 17 h → ~2–3 h)

Goal: remove the measured bottlenecks **inside the current SB3 stack** without changing the science, and land the
correctness/hygiene fixes that are cheaper now than later.

### A1. Numpy observation path in the environment *(largest single win: env 4.1× measured)*
- In `env_bracket.py`, precompute once at construction: contiguous `float32` feature matrix, `Close` array,
  `atr` array (mirroring what `_m1_high`/`_m1_low` already do), and a row-record view for the few remaining
  `row["Close"]`/`row["atr"]` accesses.
- Rewrite `_observation()` and `_position_state_features()` to gather from those arrays; eliminate all per-step
  `decision_df.iloc[i]` usage (currently ~4 row accesses × 30 µs per step) and the per-step
  `Series → astype → to_numpy` conversion (244 µs of a 275 µs step).
- Decide the fate of `history.append(dict)` per step (used only by `equity_curve()` for charts): keep for eval
  episodes, but make it optional/flagged for training loops, or accumulate into pre-allocated arrays. Not
  correctness-critical.
- **Verification:** golden-vector parity (asset #2) against the current implementation — identical action
  sequence must produce identical rewards/equity/trades; then re-run the benchmark suite and record
  single-proc and `SubprocVecEnv×4` numbers.
- *Effort: 0.5–1 day including parity test.*

### A2. Device & thread policy (removes the silent 15× CPU penalty; settles the GPU question)
- Pin `torch.set_num_threads(4)` (and document `OMP_NUM_THREADS`/`MKL_NUM_THREADS`) in `train_ppo.train()` when
  running on CPU; re-run the cuda-vs-cpu A/B from `ANALYSIS.md` §4.3 after A3 (n_envs changes the balance).
- Make the default **evidence-based**: whichever of `cpu(4 threads)` / `cuda` wins *after* A1+A3 becomes the
  documented default in `config.py` (keep `device` overridable). Note SB3's own MLP-on-GPU warning in comments
  so the choice is explainable later.
- *Effort: 0.5 day (measurement + config change).*

### A3. Vectorization width (`n_envs` 4 → 8–16) with explicit resource accounting
- Rationale from `ANALYSIS.md` §4.3: policy-forward + VecNormalize + IPC cost ≈ 1.19 ms per *vector* step
  regardless of how many envs ride in it — widening the batch amortises the fixed cost.
- Work items: sweep `n_envs ∈ {4, 8, 16}` × `batch_size ∈ {512, 1024, 2048, 4098}`; confirm
  `effective_buffer = n_steps × n_envs` and minibatch divisibility warnings; **RAM budget per worker** (each
  forkserver worker pickles the fold's M1 slice and the env copies it again — measure RSS at 8/16 envs on a
  sliding fold, 62 GB machine but folds will later run concurrently in A4); keep `start_method` logic
  (forkserver/spawn).
- Watch item: with A1 the env gets ~4× faster, so the loop may shift back toward policy-forward cost — hence the
  sweep must be measured end-to-end (`PPO.learn` steps/s), not env-only.
- *Effort: 0.5–1 day.*

### A4. Fold-level parallelism driver (the 4–5× wall-clock multiplier)
- Today folds run strictly sequentially inside `train_sliding_walk_forward`. Add an *outer* driver that runs
  K folds concurrently (K = 4–5 given 20 cores at ~4 workers each), each writing to its own
  `models/sliding/fold_k/` directory with the existing per-fold artifact layout.
- Design constraints: no shared mutable state (each fold already has its own `out_dir`); gate/summary/aggregation
  must run *after* all folds complete and remain byte-compatible with the current CSV schemas; per-fold stdout
  must be captured to per-fold logs (today's prints interleave); a fold failure must not kill the sweep
  (record and continue, mark in summary); GPU shared across concurrent folds is fine (GPU work is a small
  fraction — verify no OOM: each PPO process allocates its own CUDA context).
- Choose implementation: independent processes launched by a thin orchestrator (recommended — isolates CUDA
  contexts and crashes) vs threads (rejected: GIL + CUDA context sharing).
- *Effort: 1–1.5 days (including failure handling and a dry-run on 3 folds).*

### A5. Data-ingress cache (removes ~4–6 s reparse × every entry point)
- One-time conversion of `data/XAUUSD_1m_EET.csv` → columnar cache (parquet/feather, f64 OHLC + tz-aware index),
  plus a cached **H1 feature frame** (137,710 × ~30 ≈ 40 MB) keyed by a hash of the feature-relevant config
  (timeframe, warmup, atr/rsi periods, date range).
- All five entry points (`run_pipeline`, `train_ppo`, `training_diagnostics`, `final_holdout_eval`,
  `view_results`) go through the loader; a `--rebuild-cache` escape hatch; cache invalidation by config hash +
  file mtime.
- *Effort: 0.5 day. (The bigger ingress play — polars/Arrow for multi-symbol/tick — is deliberately deferred to
  the Rust/ingress item in Track C's backlog.)*

### A6. Cut eval/rollout overhead
- The per-fold post-training full-history train rollout (≈110 k deterministic steps for the in-sample→OOS chart)
  costs ~1 min/fold (~35 min/sweep): make it optional (flag) or sample a window; keep it for the final/production
  fold.
- Cap `_CaptureDoneWrapper` history growth on long rollouts (ties into A1's history decision).
- *Effort: 0.5 day.*

### A7. Correctness & hygiene fixes bundled with A (cheap now, expensive later)
1. **Truncation bootstrap**: set `info["TimeLimit.truncated"]` when `truncated` fires in `env_bracket.step`
   (SB3's `DummyVecEnv` already stores `terminal_observation`) so GAE bootstraps at the arbitrary 2048-step
   boundary instead of treating it as a true terminal. Re-run one fold to confirm no metric regression.
2. **Block walk-forward mismatch**: `train_walk_forward` currently evaluates the *final* weights under the
   *best-checkpoint's* VecNormalize (`train_ppo.py:837–857`). Route it through the same
   `_load_fold_model` path the sliding scheme uses.
3. `.gitignore` (`data/`, `.venv/`, `models/`, `outputs/`, `__pycache__/`), pin `requirements.txt` to the
   versions actually in the venv, add a `tests/` directory with the A1 parity test + a leakage test + a
   split-boundary test (embargo, `test_frac` identity between schemes), fix the `train_ppo.py:517` encoding
   artifact.
4. Optional-but-cheap: cap `total_timesteps` sensitivity run (config already flags ≈96 passes/fold) — one
   1 M vs 3 M comparison fold to know whether the default is over-trained.

### Track A — Definition of Done
- Benchmarks recorded (suite from asset #3): env-only, vec-env, end-to-end, on the *new* default device policy.
- Full sliding sweep completes **≤ 3 h** on the laptop with all gates/artifacts identical in schema to today.
- Parity test green; A7.1/A7.2 fixes verified with a 1-fold before/after comparison; hygiene items merged.

---

# Track B — JAX port (1–3 weeks; the strategic core)

Goal: move the environment *and* the PPO loop into a single `jit`ed, `vmap`-batched pipeline so thousands of
environments run resident on the GPU with zero per-step host involvement — then use that headroom for swarms
(seeds × folds × hyperparameters) instead of just one faster run.

### B0. Environment smoke test *(gate before any design work; 30–60 minutes)*
- Install `jax[cuda13]` in a fresh venv; verify on **this machine** (RTX 5070 Ti Laptop, **sm_120**, driver
  590.48 / CUDA 13.1): device visible, a small `jit` matmul correct, a trivial `vmap` over 10³ elements runs.
- Known risks to test explicitly (from the scan): Blackwell/driver-590 quirks (open reports of
  "No visible GPU devices" on this exact driver with a cuda12 plugin; a jit-cache driver bug on sm_120 in
  jax 0.10.x; "No reference output" compile errors on older jax).
- Fallback ladder, decided in advance: (1) latest `jax[cuda13]` wheels, (2) CUDA 12 wheels, (3) NVIDIA's JAX
  container (26.09+), (4) driver update, (5) CPU-only JAX (acceptable for *development parity*, not for the
  target throughput).
- **Go/no-go:** if none of (1)–(4) yields a working GPU after ≤ half a day, pause B and revisit — do not design
  against an unverified accelerator.

### B1. Design decisions (write these down before coding)
1. **API:** implement against **gymnax 1.x's six-value step** (`obs, state, reward, terminated, truncated,
   info`) with auto-reset and `info["final_observation"]`; this also eliminates the current missing-truncation-
   bootstrap issue by construction.
2. **Precision:** enable `jax_enable_x64` for the environment path (money math), with an explicit decision record
   on whether the *policy network* runs f32 (recommended) while the sim runs f64 — or full f64 parity first,
   optimise later. The parity harness (asset #2) is the arbiter.
3. **State layout:** environment state as a pytree of *batched scalars* per env (episode index `i`, equity,
   realized PnL, position fields: direction/entry/sl/tp/units/risk_cash/tp_r/bars_in_trade, steps counter) —
   no per-step Python objects, no history lists (episode returns accumulated via a `LogWrapper`-style wrapper,
   matching JaxMARL's).
4. **Params vs state:** dataset-derived arrays (features `[T,25]`, closes, atr, M1 high/low or — preferably —
   the **first-touch LUT + window offsets** from asset #1) live as `EnvParams` (static, replicated), not
   per-env state.
5. **Normalisation:** running mean/variance for observations *and* reward stored inside the training state
   (the `VecNormalize` equivalent) with an explicit frozen mode for evaluation; sync semantics for checkpoint
   selection must replicate today's `_SyncVecNormCallback` behaviour (stats travel with each saved checkpoint).
6. **Action semantics:** `MultiDiscrete([3,3,4])` mapped exactly as today (direction 0=flat/1=long/2=short;
   bracket dims only effective at entry — keep the behaviour identical, do not "improve" the action space in
   the same change; revisit as a separate experiment).
7. **Episode regime:** preserve `randomize_start` with the same headroom rule and `max_episode_steps=2048`
   truncation (with proper bootstrap).

### B2. Environment implementation *(3–5 days incl. parity)*
- **Reset:** sample start index per env (vectorised RNG split), initialize state, emit observation.
- **Step:** (a) apply direction/flip/manual-close at decision close; (b) open position using entry-price
  formula; (c) intrabar outcome via the **LUT** (O(1) gathers — replaces the current per-step window scan
  entirely) with SL-first rule; (d) equity/realized update and reward exactly as `env_bracket.step` does
  (risk-unit normalisation, MtM shaping at weight 0.01, holding penalty); (e) position-state features;
  (f) increment + auto-reset on terminated/truncated inside the wrapper.
- **LUT construction:** build on the host with numpy/pandas (validated against the current scan on a large random
  sample of (bar, combo) pairs), ship as device arrays; document the fixed window width (H1 → ≤60 M1 bars) and
  the padded/gather strategy; handle non-uniform windows (gaps) with an offset-pair array + mask.
- **Verification (hard gate):** golden-vector parity vs the pandas env — obs, reward, terminated/truncated,
  equity, trade log (times, prices, exit reasons, r-multiples) across: (i) random-policy episodes at several
  seeds, (ii) hand-built edge cases (same-candle SL+TP hit, flip at decision close, entry at data end, start
  with open-position impossible, truncation boundary), (iii) one full sliding-fold deterministic rollout vs the
  current `_rollout_on_split` output.
- *Also re-run `leakage_checks.py` logic conceptually: features are precomputed on the full series exactly as
  today — no new causality surface.*

### B3. Trainer port *(3–5 days)*
- Start from **PureJaxRL's `ppo.py`** structure (or JaxMARL's baseline file if the multi-agent door is wanted
  from day one): `vmap(reset/step)` × `lax.scan(NUM_STEPS)` rollout → GAE via reverse `lax.scan` → minibatch
  epochs via nested `lax.scan`; entire `train(rng)` wrapped in `jax.jit`.
- **Hyperparameter mapping to `config.py`** (must be 1:1 or explicitly documented): `n_steps=2048` → `NUM_STEPS`
  × `NUM_ENVS` (choose 1024–4096 envs × 128–512 steps to hit ≥2048 transitions/env and similar buffer size per
  update), `batch_size`, `n_epochs=5`, `clip_range=0.1`, `ent_coef=0.03`, `vf_coef=0.5`,
  `gamma=0.99`, `gae_lambda=0.95`, `weight_decay=1e-5` (optax: AdamW-style `add_decayed_weights`),
  linear LR decay 6e-5 → 0, grad-clip, `target_kl=0.025`.
- **Deliberate deviations to resolve during design (document each):**
  - `target_kl` epoch early-stop has no clean `lax.scan` equivalent — options: (i) fixed epochs + KL-based
    *post-hoc* logging, (ii) `lax.while_loop` per-update, (iii) epoch-count schedule calibrated to match
    observed KL. Choose the simplest that preserves the *intent* (bounded policy movement), and note it in the
    config comments.
  - Optimizer state / weight decay / LR schedule equivalence: unit-test a single update against a reference
    (small numpy/torch recomputation of the same loss) rather than trusting signs.
- **Checkpoint selection:** port `_ConsistencyEvalCallback` semantics — deterministic episode on the train-tail
  slice and on val, `score = min(q_train, q_val)` with drawdown penalty, eligibility (both legs profitable,
  ≥ `min_trades`), snapshot of normalisation stats with every save. Implementation choice: run these evals as
  *separate* jitted functions driven from Python at `eval_freq` (simplest, keeps `jit` shapes stable) rather
  than inside the training `jit` (which would need `lax.cond`/checkpoint writes — defer).
- **Metric parity:** episode returns, trade counts and drawdowns from the eval rollouts must match SB3's
  numbers on the same fold within tolerance — this validates the whole chain before any speed claims.

### B4. Walk-forward driver integration *(2–3 days)*
- Keep `data_loader.make_sliding_folds`, the gate (`_passes_consistency_gate`), summaries, stitching of test
  windows, promotion/`NO_DEPLOY` logic — all unchanged, in Python.
- Replace only the inner `train()` call: push that fold's feature/M1 (or LUT) slices to device once, call
  `jit(train_fn)(rng, data)`, pull per-eval metrics and final artifacts (params as safetensors/npz + norm stats)
  into the existing directory layout (`models/sliding/fold_k/…`), and write a `run_info.json` that
  `final_holdout_eval.py` / `training_diagnostics.py` can consume (they load SB3 `.zip` today — add a loader
  branch for the JAX checkpoint rather than rewriting those scripts).
- Evaluation rollouts (val/test windows, deterministic, full equity+trade log) run as a separate jitted function
  and feed `evaluate.py` unchanged (it's pure pandas/numpy metric code — reuse as-is).
- Sequential-fold mode first (parity of *outputs* with today), then hand the loop to the A4-style concurrent
  driver (multiple JAX processes need `XLA_PYTHON_CLIENT_PREALLOCATE=false` or fractional allocation — decide
  during B4, since one process otherwise hogs the VRAM).
- *Effort incl. one end-to-end sliding run and artifact diffing: 3–4 days.*

### B5. Throughput validation & swarms *(3–5 days, then open-ended)*
- Measure and record: env-only steps/s (jit `vmap` step at 1 k/4 k/16 k/64 k envs), end-to-end train steps/s,
  compile time per shape, host↔device volume per rollout (should be ~MB), VRAM residency.
- **JIT hygiene on Blackwell (known pitfall):** one `jit` closure per config, all varying values (LR, fold data,
  seeds) passed as *arguments*; never capture changing Python floats in closures; use `jax.clear_caches()`
  between discrete experiments; keep shapes static across a sweep.
- **Swarm layer (the actual payoff):**
  - Level 1 — batch *seeds*: leading batch dim on the trainer (or N concurrent single-seed trains with
    fractional GPU memory) to quantify seed variance (fixes the single-seed-42 weakness flagged in
    `ANALYSIS.md` §7).
  - Level 2 — batch *evaluations* across folds/windows for the gate (all 34 test windows scored in one jitted
    call).
  - Level 3 — **evosax** for population/PBT over PPO hyperparameters or policy weights (multi-device sharding
    is a one-liner in its docs), giving you the "agent swarm" capability without hand-rolling sweep infra.
- **Target:** full 34-fold × 3 M-step sweep **in minutes-to-tens-of-minutes** on the laptop GPU; rented-GPU runs
  then buy swarm width (more seeds/envs), not just clock speed.

### Track B — Definition of Done
- Parity harness green (obs/reward/equity/trades) vs the pandas env on ≥2 folds + edge cases.
- Existing gate/summary/holdout scripts consume JAX-produced artifacts without modification (loader branch only).
- Recorded numbers: env steps/s and end-to-end steps/s at ≥3 batch widths; full sliding sweep wall-clock with
  schema-identical outputs diffed against a Track-A SB3 run (metrics within documented tolerance).
- Deviations log (KL early-stop, precision policy) written into `config.py` comments or the README.

**B risks/efforts:** numerics/precision drift (mitigated: x64 + parity harness), auto-reset semantics
(mitigated: gymnax 1.0 API + final_observation), Blackwell quirks (mitigated: B0 fallback ladder + JIT hygiene),
metric mismatch from normalisation differences (mitigated: eval-parity step in B3), version churn in
gymnax/JaxMARL pinning (pin everything).

---

# Track C — cutile / cuda-oxide spike (timeboxed 2–4 weeks; parallel prong, not on the critical path)

Goal: learn the new Rust/CUDA surfaces *on a real, bounded problem* — the bracket env-step kernel — and decide
whether they earn a place in the stack. Explicitly **research**: immaturity accepted, abandonment cheap.

### C0. Toolchain verification spike *(1–2 days, hard go/no-go)*
- **cutile:** install CUDA **13.2/13.3 toolkit** alongside the existing 13.1 (set `CUDA_TOOLKIT_PATH` in
  `.cargo/config.toml`); Rust ≥ 1.89 stable; `cargo add cutile`; run the shipped vector-add/persistent-GEMM
  examples; confirm a kernel for **sm_120** compiles (Tile IR 13.2) and *loads* under **driver 590**
  (cubin path should be fine under CUDA-13 minor-version compat ≥580 — verify, don't assume).
- **cuda-oxide:** pinned Rust 1.99.1 + `rustc-dev` components, LLVM 21 with NVPTX, `cargo oxide doctor`,
  `cargo oxide run vecadd --arch sm_120`; verify the emitted PTX target is one driver 590 understands
  (their docs' PTX 9.3-vs-driver-580 failure mode is the exact trap to test).
- **Go/no-go criteria:** both examples run on this laptop within the timebox, *or* at least one runs (if only
  cutile runs, the spike proceeds tile-only; if neither, C is archived with a written note and Track B remains
  the sole GPU path).
- Also record: install steps into a `toolchain.md` (this is half the value of the spike).

### C1. Kernel design *(2–3 days of design before writing kernels)*
- **Formulation T (cutile/tile):** batch dimension maps to tiles; per-env work = gather feature row → apply
  action → LUT lookups → arithmetic state update → store obs/reward/done. Leverage `partition` on the mutable
  output tensors; static tile shapes (start `[128]`/`[256]` per the performance guide); dynamic read-only inputs
  where shapes vary; f64 supported — decide f64 state vs f32 policy tensors (note: consumer Blackwell f64
  *FLOP* rate is heavily cut, but this kernel is gather/memory-bound, not FLOP-bound — measure rather than
  assume).
- **Formulation S (cuda-oxide/SIMT):** one thread per environment episode — the natural fit for the branchy
  state machine (flat/in-trade, SL/TP/flip branches); `DisjointSlice` for outputs, `launch_contract` for
  geometry; `#[launch_bounds]` sized to occupancy.
- **Common:** SoA state arrays; dataset + LUT (asset #1) resident in device memory for the whole run; batch
  sizes 2¹⁴–2²⁰ envs; step contract = (actions in, obs/reward/done/state out), state persists across launches
  (no host round-trip); reset kernel alongside step kernel; **scope cut explicitly: no policy, no PPO, no
  normalisation in the spike.**
- Write the numerical contract first: same f64 order-of-operations as the reference, same SL-first rule, same
  sentinels — then the C5 parity test has teeth.

### C2. Host harness & raw measurement *(2–3 days)*
- A small Rust binary (cudarc/`cuda-core` streams, `cuda-async` DeviceOps for cutile): load LUT + features,
  allocate state, launch K steps × N envs, time it (excluding/include compile separately — cutile JIT-caches
  cubins), report steps/s and compile time; repeat for both formulations if both cleared C0.
- First honest metric: **raw kernel steps/s vs the numpy-obs env (14,886/s) vs a single-process ceiling.** This
  number alone justifies or kills the spike (target in the gate below).

### C3. Trainer-seam decision *(2–3 days, after C2 numbers exist)*
Evaluate in this order (each option documented with measured numbers):
1. **Burn/CubeCL trainer** (all-Rust): Burn 0.20 has Blackwell matmul paths; CubeCL docs state cuTile PTX can be
   executed by its CUDA runtime via cudarc — i.e. a Burn trainer *plus* our cuTile kernel in one process is a
   documented integration, though **we are the ones gluing it** (tensor → `CUdeviceptr` → kernel → back).
   Uses existing Rust RL crates (`rl4burn`/`thrust-rl`/`r2l`) for PPO scaffolding; risks: pre-alpha APIs.
2. **JAX trainer + kernel bridge** (if Track B is already green): keep everything else in JAX; the *Python* cuTile
   DSL exposes `cuda.tile.jax.cutile_call()` (JAX FFI) — note this is a different front-end (Python, not Rust)
   sharing Tile IR with cutile-rs; so this seam means either re-expressing the kernel in cuTile-Python, or
   building our own FFI from the Rust kernel. Honest framing: best end-to-end story, weakest Rust continuity.
3. **PyO3 zero-copy → keep Python trainers** (SB3 or `sbx`): easiest interop, but inherits the measured SB3
   per-step ceiling — only acceptable if combined with PufferLib-style async sampling. Likely a dead end for
   *throughput*, useful only as an integration reference.
4. **Hand-rolled PPO in Rust on raw cutile kernels:** rejected unless (1) proves too painful — re-deriving
   autodiff/GAE/Adam is where spikes die.
- **Output:** a one-page decision record (numbers, recommended seam, effort estimate) — *not* a commitment to
  execute it.

### C4. Parity & documentation *(1–2 days)*
- Replay golden vectors (asset #2) through the GPU kernel vs the reference numpy/pandas env: equity, trade log,
  exit reasons; document tolerances (f64, ordering).
- Write up: measured throughput, compile/JIT behaviour, toolchain repro, lessons per surface (tile vs SIMT —
  which felt better for a branchy simulator), API-breakage encounters, and a recommendation:
  **adopt (with seam plan) / archive (JAX remains the GPU path, Rust stays CPU-side for ingress/env).**

### Track C — Definition of Done (timebox respected)
- Either: **kernel runs on this laptop at ≥10× the numpy env (>150 k steps/s raw)** *and* a trainer seam is
  chosen in writing — or: written archive note with toolchain repro + why JAX remains the GPU path.
- No Track-B deliverable depends on C at any point; C never blocks A or B.
- Total timebox: **≤ 2 weeks elapsed for C0–C2, ≤ 4 weeks including C3/C4**; weekly review, abort allowed at
  any gate.

**C risks:** alpha APIs/renames (cutile "expect bugs and API breakage"; cuda-oxide early alpha + pinned
toolchain), CUDA 13.3 toolkit coexistence with 13.1 on the laptop, PTX/driver mismatches (driver 590 is CUDA
13.1 — cubin path safer than PTX path), name-collision footgun (crates.io `cuda-oxide` is a *different* 2021
GPL crate — install NVIDIA's from git), zero RL precedent on either surface (you are the pioneer), opportunity
cost (hence the hard timebox), and the integration burden: neither surface ships autodiff, optimizers, normalisation,
checkpointing or distributed training.

---

## Cross-track summary

| Track | Effort (elapsed) | Primary deliverable | Hard gate |
|---|---|---|---|
| **A — Python quick wins** | 3–5 days | Sweep 17 h → ≤3 h; parity test; hygiene/correctness fixes | Full sliding sweep green + benchmarks recorded |
| **B — JAX port** | 1–3 weeks | GPU-resident env + PPO, swarms, same harness/artifacts | B0 GPU smoke **and** golden-vector parity + artifact-compatible sweep |
| **C — cutile/cuda-oxide** | ≤2 wks (C0–C2), ≤4 wks total | Toolchain repro + measured env kernel + seam decision record | C0 examples run on sm_120/driver 590; ≥10× numpy-env or archive |

**Shared assets (§ top) are the load-bearing investment:** the LUT spec, the golden-vector parity harness and the
benchmark suite let all three tracks claim (or refute) progress with the same yardstick, and let the project
switch engines without invalidating any walk-forward result produced under the previous one.

**Immediate next actions:** (1) A1 parity test + numpy observation path; (2) B0 JAX GPU smoke test on
sm_120/driver 590; (3) C0 cutile example under CUDA 13.3 — three parallel, independently-gated first steps.
