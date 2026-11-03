# SCALING_GPU.md — Framework scan: scaling the RL engine (agent swarms, GPU, Rust/CUDA)

**Date of scan:** 2026-11-01
**Scope:** *Framework-first* review (as requested — the engine, not the strategy). We assume the bracket-trading
`BracketTradingEnv` + walk-forward harness from `ANALYSIS.md` and ask: **what technology stack should carry it to
10²–10⁶× environment throughput, agent swarms, and short/moderate rented-GPU runs?**

Sources: local clones `../JaxMARL` (bold-lab-ai/JaxMARL, tip 2026-10-26) and `../JaxMARL-HFT` (vmohl/JaxMARL-HFT,
tip 2026-05-26), `../burn` (0.20.0-pre.4, 2025-12-01), the JaxMARL-HFT ICAIF'25 paper (arXiv 2511.02136),
PufferLib papers/blogs, NVIDIA's cuTile/cuda-oxide releases, crates.io/PyPI/GitHub state as of the scan date, and
benchmarks I ran locally (documented in `ANALYSIS.md` §4).

---

## 0. The performance envelope we are escaping

From `ANALYSIS.md` (measured on this laptop: 20 cores, RTX 5070 Ti Laptop **sm_120**, driver 590.48 / CUDA 13.1):

| Path | Throughput | Note |
|---|---|---|
| `BracketTradingEnv` alone (as written, pandas obs) | 3,640 steps/s | 244 µs/step is the pandas observation |
| Same env, numpy-precomputed obs | 14,886 steps/s | 4.1× — lowest-hanging fruit still unclaimed |
| `SubprocVecEnv×4 + VecNormalize` | 5,198 steps/s | main-process loop now dominant |
| **PPO.learn end-to-end (SB3, 4 envs, cuda)** | **1,886 steps/s** | SB3 per-step `predict` + VecNormalize + IPC ≈ 1.19 ms/vec-step fixed cost |
| PPO.learn end-to-end (SB3, cpu, 4 torch threads) | 2,649 steps/s | beats GPU for MLP policy (SB3 warns about this) |
| Live default sweep (34 folds × 3 M steps = 102 M) | **≈ 17 h** | strictly sequential folds |

**Implication:** even a *perfect* environment inside SB3 caps around a few thousand steps/s, because SB3's control
loop is Python per step. Any serious scale-up must replace **the loop**, not just the env. That is the real decision
being made here.

---

## 1. What any replacement must provide (derived from the codebase)

1. **A custom environment** with: `MultiDiscrete([3,3,4])` actions, 31-float observations, an intrabar TP/SL
   simulation against a *fixed* in-memory M1 dataset (7.87 M bars / 495 MB), f64 price math, risk-normalised reward,
   `terminated`/`truncated` semantics, random start points.
2. **On-policy PPO** with GAE, advantage normalisation, entropy bonus, weight decay, LR schedule, target-KL early
   stop, running observation normalisation (the `VecNormalize` equivalent) and best-checkpoint selection.
3. **Massive env batching** — the whole point: 10³–10⁶ parallel episodes, ideally resident on the accelerator with
   zero per-step host↔device traffic.
4. **Interoperation with the existing Python harness** — walk-forward folds, the consistency gate, `evaluate.py`
   metrics, Plotly artifacts, the sealed-holdout flow. This is *not* worth rewriting.
5. **Sweeps**: 34 folds × seeds × hyperparameters = the actual research workload; embarrassingly parallel, and the
   cheapest thing to spread over rented GPUs.
6. **Multi-GPU path later** (single job >1 GPU, or one job per GPU).

**Structural gift:** the market data is fixed and brackets are *discrete* (3 SL × 4 TP × 2 directions = 24 combos).
For every decision bar `i` and combo `c`, the SL/TP prices are deterministic (`entry = f(close[i], spread,
slippage)`, `sl = entry ± mult·atr[i]`, `tp = entry ± tp_r·sl_dist`). The first-touch index inside the following
M1 window can therefore be **precomputed once** into a lookup table of shape `[n_bars, 24, dir]` (~13 MB as int16).
The entire env step then collapses to O(1) gathers + arithmetic — no window scan at rollout time. This makes the
env ideal for *any* vectorized backend (JAX `vmap`, CUDA kernel, SIMD over SoA arrays) and is worth ~an order of
magnitude on its own. (Same precompute also serves a Rust or PufferEnv implementation — it is backend-agnostic.)

---

## 2. The landscape (scan results)

### 2.A JAX full-stack: environment **and** training loop on the accelerator

| Project | State (2026-11) | What it gives us | Fit |
|---|---|---|---|
| **gymnax** 1.0.1 (2026-10-18) | Maintained; supports JAX 0.6–0.11, Gymnasium 1.1–1.3 | The *env API* to implement against: functional `reset/step` returning `(obs, state, reward, terminated, truncated, info)`, auto-reset, PyTree obs. Docs: CartPole at **0.05 s / 1M steps on A100** (~20 M steps/s) for simple envs. Also ships PPO + ES baselines. | **Core** — note the 6-value API natively fixes the missing-`TimeLimit.truncated`-bootstrap bug in our env |
| **PureJaxRL** (luchris429) | 1.1 k★, active-ish; Apache-2 | Single-file `ppo.py` template: `vmap` envs × `lax.scan` timesteps × minibatch epochs, all `jit`ed. README claim: **>1000× vs typical PyTorch RL** with many parallel agents; explicitly markets "tons of seeds in parallel on one GPU" | **Core** — this is the ~300-line trainer we port to |
| **JaxMARL** 0.2.0 (local clone) | Active (bold-lab-ai/FLAIROx; docs foersterlab.com) | Multi-agent envs + wrappers (`LogWrapper`, `JaxMARLWrapper`, `CTRolloutManager`, `GymnaxToJaxMARL` adapter) and CleanRL-style baselines (IPPO/MAPPO/QMIX…, "based on PureJaxRL") | **Optional**: for a *single-agent* env JaxMARL itself adds little beyond conventions; it becomes valuable **if** you later want a swarm of cooperating agents (e.g. per-timeframe or maker+taker agents) — its dict-of-agents API is exactly that |
| **JaxMARL-HFT** (local clone, ICAIF'25 paper) | 70★, paper-published, Apache-2 | **The direct domain reference**: JAX limit-order-book env (JAX-LOB) + IPPO-RNN trainer, `vmap` over `NUM_ENVS` (512–4096) × `lax.scan` over `NUM_STEPS`, Hydra/W&B, and a **`ippo_rnn_JAXMARL_pmap.py` multi-device variant** (`N_DEVICES`, 8× L40S in the paper). Measured (paper, single GPU, 4 000 envs, random policy): **21,969–351,119 env steps/s** for LOB replay vs 1,805–4,896 for a 64-core CPU MARL env; **5×–240× end-to-end** training speed-up vs CPU frameworks | **Template** — a far heavier env than ours already hits 10⁴–10⁵ steps/s on one GPU; our O(1)-gather env should land 10⁶–10⁷ |
| **Mava** (instadeepai) | Active (commits through 2026-09); 938★ | Distributed MARL in JAX with **two architectures: Anakin** (end-to-end JIT, JAX envs) and **Sebulba** (actors on CPU envs + accelerator learner, for non-JAX envs); `pmap` scaling; Sable algorithms | **Reference** for the distributed pattern (and the Sebulba fallback if we ever keep a Python env) |
| **evosax** 0.3.2 (2026-08) | Maintained, production-ish | 30+ evolution strategies, `ask/eval/tell`, one-line multi-device sharding (`Mesh`/`NamedSharding`), `BraxProblem`-style vectorized fitness | **The "agent swarm" primitive**: ES over policies/seeds/bracket-archs with population sharded across GPUs; also hyperparameter/PBT sweeps |
| **sbx** (araffin — SB3's author) 0.28.0 | Active | SB3-compatible API in JAX (PPO/SAC/…), RL-Zoo compatible | **Caveat read from source:** `sbx.PPO` inherits SB3's *Python* `collect_rollouts` (only `_one_update` is jitted JAX) ⇒ **it keeps the exact per-step bottleneck we measured**. Useful as a low-friction API shim, not as a throughput answer |

**JAX verdict:** this is the only mature, one-language, end-to-end GPU-resident path with a proven domain
precedent (JaxMARL-HFT) in *this* problem family. The whole pipeline — env, policy, GAE, updates, normalisation —
compiles into one `jit`ed function with no per-step CPU involvement, which is precisely the failure mode of the
SB3 loop measured in `ANALYSIS.md`.

### 2.B Stay-in-Python, replace the *loop* (native vectorization / async sampling)

| Project | State | What it gives us | Fit |
|---|---|---|---|
| **PufferLib** 2.0→3.0 (RLC'25 paper "1 M steps/s"; 3.0 blog "4 M sps on an RTX 5090"; current docs describe a ~10 k-line **CUDA C** native backend, "PuffeRL up to 60 M steps/s") | Very active (daily posts as of 2026-03), MIT, 6 k★ | One-line wrappers for arbitrary Gymnasium envs; **async/double-buffered sampling** (EnvPool-style: run M≫N envs, always take the first N ready → policy never waits on the slowest env, drives GPU idle to 0); vectorized throughput typically 2–10× vs SB3/Gymnasium on their tables (env-dependent,
with outliers either way), headline 1 M-sps native envs and 300 k–1.2 M sps training on one GPU; **`PufferEnv` C API** with starter template for writing your own env natively; multi-GPU sweeps built in (`puffer sweep --sweep.gpus=4`); "Protein" auto-tuning | **Best "no rewrite" option**: keeps SB3/CleanRL-style Python, kills the sync-wait cost. A *native* PufferEnv port of the bracket env is also viable (the C API is a Gymnasium analogue) |
| **EnvPool** (NeurIPS'22, sail-sg) | Mature but narrowed (C++ envs: Atari/MuJoCo; async pool to 1 M+ FPS on DGX) | The original C++ batched pool; ~2.8× vs Python subprocess on a laptop | **Not directly usable** (no custom-env path for our simulator) but is the design PufferLib/PufferPool reimplements |
| **Sample Factory 2.0** | Effectively frozen (still pins `gymnasium<1.0`, `numpy<2`) | Async actor-encoder-decoder, 100 k–1 M sps class numbers | Reference only; aging deps |
| **Ray RLlib / TorchRL** | Active | Distributed orchestration; TorchRL adds TensorDict overhead (rlox benchmarks: TorchRL SerialEnv ~150× slower than Rust env) | RLlib only if you want cluster job scheduling; not a throughput fix for this env |

**Stay-in-Python verdict:** PufferLib's async pool is the single highest-leverage, lowest-risk change *after* the
numpy-observation fix — it attacks the measured 1.19 ms fixed per-vector-step cost directly. It also composes with
a Rust/C env later (PufferEnv C API) if you want the native track without leaving the Python harness.

### 2.C Rust — the 2026 ecosystem (much further along than "RL in Rust" suggests)

**RL trainers/libraries:**

| Crate | State | Notes |
|---|---|---|
| **r2l** 0.0.3 | Pre-alpha (2025+) | PPO/A2C builders; **both Candle and Burn backends**; **Gymnasium adapter** (Python env ↔ Rust trainer); multi-categorical (i.e. MultiDiscrete) policies; obs/reward normalisation; single- and multi-threaded rollouts |
| **thrust-rl** 0.4.0 (2026-06) | Published, fast-moving | Burn-backed (CUDA/ROCm/Metal/Vulkan/WebGPU/NdArray); PPO/A2C/DQN/SAC/BC + LSTM-PPO with V-trace, **async actor-learner**, multi-agent PSRO/NFSP, WASM export; explicitly "inspired by PufferLib" |
| **rl4burn** 0.1.1 (2026-03) | Crates.io | Burn; **PPO discrete/masked/MultiDiscrete/continuous** (matches our `MultiDiscrete` action space) + DQN |
| **rwrd** | Early | SB3-aligned API (`learn`/`predict`/`save`/`load`) on Burn |
| **brand** (e3ntity) | Early dev, not on crates.io | Backend-agnostic (Burn/Candle); "fearless concurrency for vectorized environments" — the right design idea, too immature to bet on |
| **rlox-candle** 0.2.0 | Crates.io | Candle backend + "hybrid Rust-side data collection" (Rust vec env feeding a trainer) — the architecture we'd want |

**Environments/vec-envs (Rust):** `operant` (SIMD/SoA Gymnasium envs, PyO3 zero-copy numpy, claims **97.5 M steps/s** on 4 096 CartPoles ≈ 600× Gymnasium), `gmgn` (faithful Rust Gymnasium port + `SyncVectorEnv`/`AsyncVectorEnv`), `rlox` (Rayon `VecEnv`, benchmarks: **292 ns/step native, 2.7 M steps/s @ 512 envs**, Gym bridge overhead only 2.6 µs, vs SB3 Subproc 163× slower at 1 env).

**Trainer backends:** local `../burn` 0.20.0-pre.4 has `burn-cuda`, `burn-wgpu`, `burn-rocm`, `burn-candle`,
`burn-tch`, **`burn-cubecl`**; Burn 0.20 (2026-01-15 release blog) added CubeK launch-latency work and **Blackwell
(RTX 5090) TMA + PTX-MMA matmul paths matching CUTLASS/cuBLAS**. **CubeCL** (tracel-ai) is a Rust `#[cube]` kernel
language JIT-compiling to CUDA/HIP/Metal/Vulkan/WGSL/CPU — alpha but "used in production by Burn", 1.85 M downloads.
Crucially its docs state: *"a kernel produced by a separate DSL like **cuTile** lowers to PTX, and the CubeCL CUDA
runtime executes PTX through **cudarc**, so external kernels can plug directly into the runtime."*

**Rust verdict:** a credible **all-Rust** stack exists today (env in Rust + PPO on Burn/Candle with CUDA), and the
env-only path (Rust vec env → PyO3 zero-copy → keep SB3/JAX) is nearly free. What does *not* exist is Rust's
equivalent of JAX's "one `jit` over everything": Rust RL is threaded-CPU-rollouts + GPU-matmul, not GPU-resident
environments. For *GPU agent swarms*, JAX (or CUDA kernels) remains mandatory; for *raw env throughput per euro*,
Rust CPU is excellent.

### 2.D The Rust/CUDA surfaces you named: `cutile` and `cuda-oxide`

NVIDIA's own framing (developer blog, 2026-09-08, "Introducing CUDA Rust: Two Tracks"):

| | **cutile (cuTile Rust)** | **cuda-oxide** |
|---|---|---|
| Model | **Tile-level** (one logical thread per tile of data; compiler maps to real threads) — dense/regular workloads, tensor cores | **SIMT** (explicit per-thread kernels, Rust→PTX via a custom `rustc` codegen backend on Pliron/LLVM) — branchy per-thread workloads |
| Crate | crates.io `cutile` **0.4.0** (created 2026-03-13, updated 2026-09-25); Apache-2.0 | GitHub **NVlabs/cuda-oxide** v0.1.0 alpha (⚠ crates.io `cuda-oxide` 0.4.0 is a *different, 2021 GPL-3* project by Protryon — name collision, install from git via `cargo oxide`) |
| Toolchain | **Stable Rust ≥ 1.89**, CUDA **13.2/13.3** (`tileiras`), sm_80+ (**sm_120 Blackwell & sm_121 GB10 documented**), Linux | **Pinned Rust 1.99.1** + `rustc-dev` components, **LLVM 21+**, CUDA 13+, Linux only, `cargo-oxide` driver |
| Maturity | Early-stage research, but "further along": finalized kernel/launch APIs at v0.1.0, **used in HuggingFace's Grout inference engine and mistral.rs**; B200 GEMM = 96.4 % of cuBLAS (2.07 PF) | **Early alpha**; docs warn of PTX/driver version traps (PTX 9.3 from CUDA 13.3 fails on driver 580 with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) |
| Types | f64 supported (`DTypeId::F64`, `ScalarType::F64`) — our money math is fine | f32/… device types incl. TMA, wgmma, tcgen05 |
| Safety | Ownership across the launch boundary: mutable tensors *partitioned* before launch, read-only shared; data-race-free by construction | `DisjointSlice` + `launch_contract`s; "no safety on GPU-side code" beyond aliasing rules |
| **What it is NOT** | **Not an RL framework**: no autodiff, no optimizers, no env API, no VecNormalize, no PPO, no distributed training | Same |

**Do they offer "an environment suited to RL"?** Not as a packaged environment — they offer a **kernel-authoring
surface**, which is the *bottom layer* of an RL stack. Concretely, for this project they could host:

1. **The batched env-step kernel** — this is a genuinely good fit. With the §1 precompute, one SIMT thread (or
   one tile) per env episode does: gather features → apply action → O(1) touch-LUT lookups → update equity. Both
   tracks handle it; `cuda-oxide`'s SIMT model matches the per-episode state machine naturally (divergent
   in-trade/flat branches), while `cutile`'s tile model matches the vectorized-window/LUT formulation. NVIDIA
   ships a **working interop example** (`cutile_inter_kernel`): cutile tile kernel + cuda-oxide SIMT kernel, same
   process, same CUDA stream, shared device memory, chained as `DeviceOp`s.
2. **The policy forward/backward** — possible but wasteful: a 128-64 MLP's PPO update is ~4 GEMMs + elementwise +
   Adam, i.e. you'd hand-roll autodiff (~a few hundred lines with `mma`, or let `burn`/`cubecl` do it — and
   CubeCL can already execute cuTile PTX via cudarc, so *Burn trainer + cuTile custom kernels* is a documented
   interop, not a fantasy).
3. **A bridge back into our Python/JAX harness** — **cuTile *Python* exists** (`cuda.tile`, NVIDIA docs) with:
   - **PyTorch integration** (pass torch tensors straight into `ct.launch`, wrap in `torch.autograd.Function`
     for backward), and — the important one for us — a **JAX FFI**: `cuda.tile.jax.cutile_call()` launches
     cuTile kernels *"as a regular op inside a `jax.jit`-compiled graph"*.

   So the "best of both" hybrid is explicitly supported by NVIDIA: **keep JAX PPO (gymnax/PureJaxRL) and write
   only the hot env kernel in cuTile**. Note this uses the *Python* DSL; for the *Rust* DSL (`cutile-rs`) you
   would be building your own FFI (`cuda.async`/`cuda-core` give you streams and `CUdeviceptr`s, and
   `load_module_from_ptx` accepts cuda-oxide PTX) — doable, but you become the integration point.

**Precedent outside NVIDIA:** SkyRL (NovaSky) already replaces PyTorch `ragged_dot` with cuTile kernels inside
an LLM-RL stack (benchmarked ~parity on MoE f16) — i.e. cutile's current niche is *kernels inside RL training
stacks*, not RL loops themselves. Nobody appears to have shipped an RL *environment* on cutile/cuda-oxide yet.

**Environment fit for *this* machine:** sm_120 (verified `(12, 0)` via `torch.cuda.get_device_capability`) is on
cutile's supported list (Tile IR 13.2) and cuda-oxide targets sm_80+; but the installed toolkit is **CUDA 13.1**
while cutile wants **13.2/13.3** (driver 590 ≥ 580 satisfies CUDA-13 minor-version compat, and cutile emits
*cubins*, so this will probably work — **verify with a 30-minute spike before planning around it**). cuda-oxide's
LLVM→PTX path (PTX 8.7) should also load on driver 590; its nvJitLink path may not (see their PTX-9.3 note).

**Rust/CUDA verdict (serious, not dismissive):** these are the right *long-term* primitives for a
GPU-resident simulator written in Rust, and the interop story (tile↔SIMT, cuTile↔JAX FFI, cuTile↔CubeCL/Burn) is
better than I expected 6 months ago. But as of today they are **kernel toolkits without an RL stack**: adopting
them for this project means *you* write the env kernel *and* own either (a) a hand-rolled PPO for a 2-layer MLP,
(b) a Burn/CubeCL trainer with a custom-kernel seam, or (c) a JAX/PyTorch trainer + cuTile-kernel bridge. Each is
a research spike, not a drop-in. Also: cutile's README still says "expect bugs, incomplete features, and API
breakage" — you said you don't mind immaturity bumps for Rust progress, which makes this a **good sandbox**, but
it should be a *parallel* track, not the critical path.

### 2.E Out of scope (checked for completeness)

LLM-RL systems (verl, AReaL, SkyRL, slime, OpenRLHF) scale multi-node GPU training beautifully but target
token-level on-policy/off-policy LLM updates, not Gym-style environments — relevant only as evidence that cuTile
is already being consumed from RL stacks (SkyRL). Classical GPU physics sims (brax, Isaac Lab, mujoco_playground)
are irrelevant to a market-data simulator, though brax/Isaac demonstrate the "env on accelerator" thesis.

---

## 3. Option matrix

| Option | Est. steady-state throughput | Port effort | Maturity | Keeps Python harness? | Multi-GPU | Risks |
|---|---|---|---|---|---|---|
| **0. Python quick wins** (numpy obs, `n_envs` 16, thread cap, fold parallelism) | 10–30 k steps/s; sweep 17 h → 1.5–3 h | Days | Fully mature | Yes | Fold-per-GPU only | none |
| **1. PufferLib async pool + (optional) native PufferEnv** | 50 k–1 M steps/s (async pool + C env) | Days–2 weeks | Active; version churn (3.0 pins `gym==0.23`, docs already describe 5.0) | Yes (SB3/CleanRL-compatible) | `puffer sweep --sweep.gpus` | API churn; C env = second implementation |
| **2. JAX port (gymnax env + PureJaxRL trainer; JaxMARL wrappers; JaxMARL-HFT as reference)** | **10⁶–10⁷ steps/s class** (anchors: gymnax 20 M/s simple envs on A100; JaxMARL-HFT 22 k–350 k/s for a *much* heavier LOB env) ⇒ full 102 M-step sweep in minutes | **1–3 weeks** (env ~200 lines + trainer ~300 + parity tests) | Mature core (gymnax 1.0.1, optax/flax/distrax); JAX-Blackwell quirks to verify | Yes (outer driver stays Python) | **Native** (`pmap`/sharding; JaxMARL-HFT's `N_DEVICES` pattern + paper's ≤8×L40S setup; evosax multi-device) | sm_120/driver-590 JAX quirks; you must re-derive normalisation + tests |
| **3. All-Rust stack** (Rust env + r2l/thrust-rl/rl4burn on Burn-CUDA; PyO3 zero-copy for eval) | 10⁵–10⁷ env steps/s on CPU threads (LUT-based step ≈ sub-µs; rlox/operant anchors) + GPU matmuls for the MLP | 2–4 weeks | Pre-alpha trainers; Burn/CubeCL mature-ish (alpha, production in Burn) | Partially (metrics/plots via saved policies) | Fold/seed parallelism; Burn multi-GPU is real but newer | Trainer APIs move; you re-implement normalisation/checkpoint conventions |
| **4. Rust/CUDA spike: env kernel in `cutile`/`cuda-oxide`** (+ Burn/CubeCL, or cuTile-Python **JAX FFI**, or hand-rolled PPO) | 10⁶–10⁸ kernel-steps/s theoretical; end-to-end depends on which trainer you pair | Weeks (timeboxed spike: 1–2 wks for env kernel only) | **Early research/alpha**; no RL precedent | Only if bridged (JAX FFI / PyO3) | Manual (multi-GPU hand-rolled) | Toolkit 13.3 install, API breakage, you own integration; **greatest learning value for Rust** |
| **5. Hybrid**: JAX trainer + cuTile env kernel via **JAX FFI** (or JAX env entirely) | Same as 2, with custom-kernel upside | 2–4 weeks | NVIDIA-documented mechanism, but young | Yes | Yes (JAX side) | Two-toolchain friction; cuts both ways on debuggability |

---

## 4. Recommendation (staged, with gates)

**North star:** keep the *harness* (walk-forward, gate, metrics, sealed holdout — `ANALYSIS.md` found it sound),
replace the *engine*, and buy swarms (seeds × folds × hyperparameters) rather than raw single-run speed.

**Stage 0 — Python floor (this week, do regardless).**
Numpy observation path in `env_bracket.py` (4.1× measured), `torch.set_num_threads(4)`, `n_envs` 4→16, parquet
cache for the CSV, parallel fold driver. Gets the sweep to ~1.5–3 h and de-risks everything downstream.
*Gate:* parity test + `train_sliding_walk_forward()` runs green.

**Stage 1 — JAX port (recommended core, ~1–3 weeks).**
1. Implement `BracketEnv` as a **gymnax-style functional env** (state as a pytree; six-value step API — this also
   fixes the truncation-bootstrap bug for free). Precompute the **first-touch LUT** (`[n_bars, 24]` per direction)
   on the host with the existing pandas code, ship as device arrays. f64 via `jax_enable_x64` (or document an
   f32 parity tolerance) — money math deserves the explicit choice.
2. Port `PureJaxRL/ppo.py` (or JaxMARL's `baselines/MAPPO` if you already want the multi-agent door open):
   `vmap(env.reset/step)` × `lax.scan(NUM_STEPS)` × minibatch epochs; add running obs normalisation
   (VecNormalize equivalent) inside the jit and a `LogWrapper` for episode stats.
3. **Differential tests**: replay identical action/seed sequences through the pandas env and the JAX env; assert
   equity/trade parity. This is the gate that protects your walk-forward comparability.
4. Keep the Python driver: for each fold, push slices to device, `jit(train)`, pull metrics into the existing
   gate/summary code. Then add **swarms**: batch seeds (and, for evaluation, folds) as a leading batch dim;
   `evosax` for population/ES sweeps if PPO-seed variance turns out to dominate.
   *Expected:* env throughput 10⁶-class (anchors above), full 102 M-step sweep well under an hour on the laptop
   GPU; a rented 4090/H100 buys more envs (thousands → hundreds of thousands resident), not merely "faster".

**Stage 2 — rented cluster (when a single laptop sweep is still too slow).**
Cheap first: **job-level parallelism** — one (fold × seed × hparam) job per rented GPU; our data is 495 MB and
trivially shipped; no code changes beyond a launcher. Then single-job multi-GPU: JAX `pmap`/sharding following
`JaxMARL-HFT/.../ippo_rnn_JAXMARL_pmap.py` (`N_DEVICES` pattern) — the paper evaluates on up to 8× L40S and
reports up to ~50× over the CPU baseline when the whole node is used. Only go multi-host (`jax.distributed`) if one job exceeds GPU-hours. (Note: JaxMARL-HFT ships a
`Dockerfile_GB10` — DGX Spark/GB10-class retail boxes are a supported target for both JAX and cutile `sm_121`.)

**Stage 3 — Rust track (parallel, timeboxed, for the learning + long-term option).**
- **3a (low risk, immediate):** data ingress in Rust (`polars`/Arrow) as the Python-facing cache, per
  `ANALYSIS.md` §8; optionally a PyO3 zero-copy `VecEnv` for the bracket env following the `operant`/`rlox`
  pattern (SoA + LUT + Rayon) — measure it against Stage 0. If you keep SB3, PufferLib's async pool (Stage 1-B)
  is the better companion than more Rust.
- **3b (the actual cutile/cuda-oxide spike, 1–2 weeks timebox):** write **just the env-step kernel** —
  precomputed-LUT version, one thread/tile per env — first on **cutile** (stable Rust, crates.io, f64, sm_120
  documented), then optionally the same step in **cuda-oxide** SIMT to compare the two tracks. Decide the trainer
  seam *after* the kernel works: (i) Burn/CubeCL (documented cuTile-PTX interop), (ii) PyO3 → keep Python
  trainer, or (iii) if you're JAX by then, validate whether cuTile-Python's `cutile_call` FFI can express the
  kernel you prototyped in Rust (the Rust DSL and the Python DSL share Tile IR, but that is *your* bridge to
  build). **Gate:** kernel beats the numpy-obs env by ≥10× *and* runs on this sm_120 laptop under driver 590
  after a toolkit 13.2/13.3 install; otherwise archive the branch and keep JAX as the GPU path.
- Do **not** attempt a hand-rolled cutile PPO unless 3b succeeds and Burn's seam feels limiting — re-deriving
  autodiff/Adam/GAE is where projects die.

---

## 5. Direct answers to the questions you asked

- **JaxMARL specifically?** Right *family*, and JaxMARL-HFT is the best in-domain reference available (measured
  22 k–351 k env steps/s on one GPU for a far heavier simulator, multi-GPU via `pmap`). But for a
  *single-agent* env the load-bearing pieces are **gymnax (env API) + PureJaxRL (trainer)** — JaxMARL itself is
  wrapper/baseline sugar you'd adopt *if* you want the multi-agent/agent-swarm-of-agents framing later.
- **Other equivalent scalable tech?** Ranked by fit: **PureJaxRL+gymnax** (core), **JaxMARL-HFT** (reference),
  **Mava** (Anakin/Sebulba distributed patterns), **evosax** (ES swarms), **PufferLib 2.0/3.0** (async sampling +
  C-API env, stay in Python), **EnvPool/Sample Factory** (reference designs), **sbx** (API shim only — keeps the
  Python bottleneck), Rust trainers **r2l / thrust-rl / rl4burn / rwrd** (real but pre-alpha).
- **cutile and cuda-oxide — seriously considered?** Yes: both are real (NVIDIA two-track CUDA-Rust announcement,
  2026-09), cutile is already in production elsewhere (HF Grout, mistral.rs), supports f64 and sm_120, and has
  documented interop with cuda-oxide, with CubeCL/Burn, and (via cuTile-Python) with **JAX FFI and PyTorch
  autograd**. They are *kernel surfaces, not RL environments* — there is no autodiff, no PPO, no vec-env
  machinery, and no RL precedent on them. The credible Rust/CUDA play here is: **env kernel first (timeboxed
  spike), trainer borrowed (Burn/CubeCL or JAX/PyTorch bridge)**, with the **all-Rust-CPU stack** as the robust
  fallback that already exists today (r2l/thrust + Burn-CUDA + Rayon envs).
- **Retail GPU cluster, later?** Yes, it scales cleanly: Stage 1 makes the workload a pure function of
  accelerator-hours; the first-order ROI is *many independent jobs* (folds × seeds × hparams) one-per-GPU, then
  JAX `pmap`/sharding for single jobs, mirroring JaxMARL-HFT's 8×GPU results. 12 GB is already enough for this
  workload locally; rented 24–80 GB cards buy env-batch width and swarm size.

**One-line summary:** *the framework is small and the data is fixed — port the environment to JAX (gymnax +
PureJaxRL patterns, JaxMARL-HFT as your in-domain template) to get GPU-resident swarms within weeks, use
PufferLib if you want the win without leaving Python, and run cutile/cuda-oxide as a timeboxed Rust/CUDA kernel
spike where you get the Rust progress without betting the pipeline on an alpha toolchain.*

---

## 6. Results & conclusions — what actually happened (Tracks A/B/C)

Everything in §§1–5 above is pre-implementation analysis. This section is the post-implementation
record: what was built, what it measured, and which predictions held. All numbers below were
measured on this machine (RTX 5070 Ti Laptop, sm_120, 12 GB, driver 590.48.01 / CUDA 13.1,
20 cores) unless noted. Decision-by-decision log: `DECISIONS.md` (Track A D1–D21/P1–P6, Track B
D22–D58, Track C D59–D64); toolchain repro: `toolchain.md`; full suite: **91 tests green**.

### 6.1 Track A — Python quick wins (the baseline that makes every comparison below honest)

| Item | Result |
|---|---|
| A1 env refactor | **~15×** (2,521–3,640 → **37,854 steps/s** random actions, 26 µs/step; 173,931/s flat; `_observation()` 244 µs → 4.3 µs = 57×). Bit-parity vs pre-refactor traces: 10 golden fixtures, obs bitwise / floats ≤1e-9 |
| A2 thread cap | `torch_threads=4` turns a thrashing sweep into a working one: per-fold **~21 → ~1,200 steps/s**, load avg **99 → 6.5** on 20 cores |
| A2/A3 device & width | **CPU beats CUDA at every width** for this MLP (5,036 vs 2,574 steps/s @ n_envs=4; 10,377 vs 6,846 @ 16); n_envs 4→16 ≈ 2×. Defaults: `cpu`, `torch_threads=4`, 4 folds × 4 envs |
| A4 fold driver | 34/34 sliding folds end-to-end (toy budget) in ~9 min, 4 subprocess workers; artifacts + gate + stitched OOS unchanged |
| A5 data cache | cold 6.57 s → warm **0.24 s (27×)**; byte-identical parquet (7.9 M M1 rows) |
| A7 correctness | truncation bootstrap (`TimeLimit.truncated`), block-WF checkpoint fix, pinned requirements |

Performance observations that shaped the design: pandas row access was **244 µs of a 275 µs step**
(vs ~1 µs for a numpy gather) — obs construction, not simulation, was the original bottleneck; SB3's
per-vector-step fixed cost (~1.19 ms, `predict` + VecNormalize + IPC) caps any SB3 pipeline at a few
thousand steps/s no matter how fast the env gets; tiny-matmul PPO on 20 cores is **15× slower** with
torch's default thread count than with 4; and GPU util under SB3 was **13–30 % at 352 MiB** — the
GPU was present but idle, which is the measured justification this whole document argued from.

### 6.2 Track B — JAX port: the predicted 10²–10⁴× lands as **187× end-to-end**

**Correctness first (the gate that everything else stands on):**
* `jax_env` (gymnax-style f64 env, padded-window gather per `LUT_SPEC.md` §3.4) — **41-test parity
  gate green on both CPU and CUDA backends**, observed trace difference **0.0 (bit-exact)** against
  the pandas env on every scenario: 10 golden fixtures, live deep-compares, 5 random seeds, 3 exact
  level-tie constructions (≤/≥ inclusivity, SL-first double touch), a full sliding-fold rollout of a
  real Track-A model vs `_rollout_on_split`, and `evaluate.full_report` metric parity.
* `jax_ppo` (PureJaxRL-structure trainer) — source-verified SB3 mappings pinned by unit tests: GAE
  vs a transcription of `compute_returns_and_advantage` (≤1e-5), one PPO update vs torch (grads
  within ~6e-4 rel — f32 reduction-order noise; optimizer chain vs an independent Adam(wd)
  transcription ≤1e-5), `RunningMeanStd` vs SB3's class (1e-12), a full VecNormalize obs/reward
  stream vs SB3's `VecNormalize` (1e-6).

**Throughput (`jax_benchmarks.py all --data real`):**

| Measurement | Result | vs SB3 / plan |
|---|---|---|
| env-only vmap step @ 1k envs | 42.3 M steps/s | plan predicted 10⁶–10⁷ class |
| @ 4k / 16k / 64k envs | 130.3 M / 162.0 M / **213.0 M steps/s** | vs SB3 pipeline 1.65–2.65 k steps/s |
| end-to-end training (rollout+GAE+update) @ 64×256 | **494,236 steps/s** | **≈187× the best SB3 rate** |
| compile per shape (rollout/gae/update) @ 64×256 | 0.96 s / 0.07 s / 0.87 s | ~2–5 s at the SB3-parity 4×2048 shape |
| trajectory volume per rollout | **2.44 MB** (host pull < 0.05 ms) | plan expected ~MB |
| VRAM residency (rollout+update) | **52 MB in use / 189 MB peak** | of 12 GB — swarm width is the remaining headroom |

**The full production sweep (the DoD wall-clock target):** 34 folds × 3 M steps (102 M env-steps +
per-fold consistency evals + val/test rollouts + per-fold JIT compiles) in **674 s ≈ 11.2 min** with
4 parallel workers — the plan's "minutes-to-tens-of-minutes" goal met. The equivalent SB3 sweep
extrapolates to **~2.8–3.5 h** (34 × 3 M at 1,886–2,649 steps/s with 4-fold parallelism); Track A
never ran it at production budget. Outputs are schema-identical to the Track-A layout (summary
columns, fold artifacts, gate, promotion, `NO_DEPLOY` semantics) via engine-aware loader branches —
proven by the B4 driver tests, not assumed.

**The sweep's own result, for the record:** stitched out-of-sample **+240.5 %** compounded, Sharpe
+0.63, max DD −36.5 %, PF 1.07 over 9,445 trades, 20/34 folds positive with PF>1 — and the
consistency gate **correctly refused deployment** (worst-fold PF 0.77 < 0.90 → `NO_DEPLOY`). The
gate logic transferred to the new engine without modification and caught the same kind of
inconsistency it exists for.

**Swarm economics (the point of the whole exercise):** 5-seed variance sweep in **75 s** (5 × 100 k
steps) — val return **−10.8 % ± 12.2 %**, Sharpe −0.35 ± 0.46 across seeds. That quantifies
`ANALYSIS.md` §7's single-seed-42 concern directly: seed variance is larger than most effect sizes
being measured, and now it is cheap enough to report routinely.

Quirk ledger (Blackwell specifics, all worked around and documented): cuSolver `Dgeqrf` (used by
JAX's orthogonal init QR) raises INTERNAL errors mid-suite on sm_120 and poisons the context →
host-side numpy QR for init (D45); torch's cuBLAS fails to initialise (`CUBLAS_STATUS_NOT_INITIALIZED`)
when JAX holds VRAM in a shared process → eval-path `PPO.load` pinned to CPU (D57); JAX's default
matmul precision is TF32 (1e-2 error on a matmul check) → `highest` for this numerical work; the
sm_120 jit-cache bug reported against jax 0.10.x did not reproduce on 0.12.0.1.

### 6.3 Track C — the spike: kernel beats the gate by ~10³, loses the seam by 2× (both predicted)

**C0 toolchain — all three surfaces pass on sm_120/driver 590** (`toolchain.md`): cuTile **Python**
(cuda-tile 1.7.0 + tileiras 13.4 via pip, no system install), **cutile-rs** 0.4.0 (`cargo add cutile`),
**cuda-oxide** (`cargo oxide run vecadd` under pinned Rust 1.99.1). The PTX-vs-old-driver trap did not
reproduce; the real friction was compiler-alpha surface (kernel scalar ABI is **f32** — scalars had to
move to buffers; `astype(int64)` hits a `trunci` verification bug; an i64 min-reduce **silently
miscompiles** and corrupts neighbouring values — the kind of finding that only a differential parity
harness catches).

**C2 raw kernel throughput** (Formulation T, f64 state, one launch per step):

| Envs/step | cuTile kernel | JAX vmap env (Track B) | numpy env (Track A ref) |
|---|---|---|---|
| 1,024 | 44.2 M steps/s | 42.3 M | ~14.9 k (plan's C2 baseline; A1 random-actions 37.9 k single-proc) |
| 4,096 | 84.9 M | 130.3 M | — |
| 16,384 | 126.7 M | 162.0 M | — |
| 65,536 | **141.5 M** | **213.0 M** | — |

**≈9,500× the numpy env** — the DoD gate was "≥10× (>150 k steps/s raw)"; met by ~3 orders of
magnitude.

**C3 seam decision record (the load-bearing question):** the kernel runs *inside `jax.jit`* through
`cuda.tile.jax.cutile_call` with bit-exact outputs (`test_jax_ffi_seam_step`) — i.e. the §4 hybrid
("JAX trainer + kernel bridge") is mechanically real. Measured on identical 128-step rollouts ×
4,096 envs: **77.7 M steps/s through the seam vs 160.4 M for XLA-vmapped `jax_env.step` — the
hand-written tile kernel is ~2× slower than the generic lowering** on this select-style, gather-heavy
step (raw kernel 141.5 M vs 213 M at 64 k tells the same story outside the graph). Verdict (D63/D64):
**keep the JAX trainer end-to-end; archive the seam; adopt the kernel + toolchain as the artifact.**
Re-entry triggers are explicit: a step-function-LUT kernel (`LUT_SPEC.md` §3.1 — gather + binary
search is where custom kernels plausibly beat generic lowering), a Burn/Rust trainer move (cutile-rs
continuity, C0-validated), or intra-kernel policy/env fusion as a goal. Formulation S (cuda-oxide SIMT)
was deliberately not implemented: it would need >2× the tile form to matter, and the env is not the
bottleneck.

### 6.4 Performance observations picked up en route (consolidated)

1. **The bottleneck moved twice.** Originally obs construction (244 µs/step); after A1 it was SB3's
   per-step Python loop (~1.19 ms/vec-step ceiling, GPU idle at 13–30 %); after the JAX port the env
   is *thousandfold* over-provisioned (213 M env-steps/s vs 494 k end-to-end) — the PPO update/rollout
   machinery is now the constraint. Env-kernel optimisation is the wrong lever at every stage so far,
   which is exactly why Track C's honest answer is "no seam today".
2. **Small-MLP RL on one GPU is a CPU workload under SB3** (CPU 2,649 vs CUDA 1,739 steps/s; per-step
   `predict` transfer ~0.9 ms CUDA vs 0.41 ms CPU) but a GPU workload under JAX (one `jit`, zero
   per-step host involvement) — the framework, not the hardware, decided where this ran.
3. **Thread hygiene beats algorithmic tuning in shared sweeps:** torch's default 20 threads made
   multi-fold sweeps 50× slower than a 4-thread cap (load 99 vs 6.5).
4. **JAX numerical hygiene:** TF32 is the matmul default (10⁻² errors); x64 money math is bit-exact
   on this GPU; cross-toolchain f64 is 1-ulp (≤2e-13 measured vs the kernel) — tolerances must be
   policy, not hope (D32/D49).
5. **One GPU, many processes:** JAX preallocates 75 % VRAM by default and breaks co-tenants
   (cuBLAS init failure) — `XLA_PYTHON_CLIENT_PREALLOCATE=false` for parallel fold workers (D52),
   CPU loads for eval-path torch (D57). Peak footprint of the whole JAX pipeline: 189 MB.
6. **VRAM is not the swarm constraint** — 52–189 MB of 12 GB in use; compile time and per-fold data
   prep are what a 34-fold sweep pays, not memory.
7. **Seed variance (−10.8 % ± 12.2 %) exceeds most effect sizes** — single-seed RL results on thin
   signals should be treated as anecdotes (now measurable in 75 s).
8. **Parity harnesses catch silent kernel corruption** that unit tests cannot: the cuTile i64-reduce
   miscompile produced plausible-but-wrong prices; only bit-level replay against the reference found
   it. Same class of bug was found in the JAX toolchain path itself (`rollout_on_split` transposed
   trade indexing — equity moved while `n_trades=0`), which is why artifact-producing paths got their
   own regression tests.
9. **Alpha-surface tax is real but bounded:** three GPU toolchains (cuTile, cutile-rs, cuda-oxide)
   were installed and verified in one session; the surprises were compiler internals, not
   environments — and every workaround is written down (`toolchain.md` §5) so the next person pays
   minutes, not days.

### 6.5 Conclusions

* **Track A (adopted, shipped):** 15× env, 27× data ingress, and the harness (golden parity, gate,
  artifacts) that made every later number trustworthy. It also established the negative result that
  justified everything else: *SB3 cannot use this GPU.*
* **Track B (adopted, shipped):** the §2 prediction ("10²–10⁴× env throughput, full sweep in
  minutes") is measured fact — 213 M env-steps/s, 494 k end-to-end (187×), 34-fold production sweep
  in 11.2 min, seed sweeps in 75 s. Parity is bit-exact against the pandas engine; the walk-forward
  gate/summary/holdout machinery is unchanged and still does its job (it refused deployment on the
  first JAX production run, correctly).
* **Track C (timeboxed spike, concluded):** the §4b gate outcome in its positive form — kernel
  ≥10× numpy env by ~10³, sm_120-under-driver-590 verified on **three** toolchains, and the trainer
  seam chosen in writing with measured numbers. The seam verdict is *no seam*: XLA's vmapped env
  beats the hand-written tile kernel ~2× here, so **JAX remains the GPU path** and the kernel,
  parity suite and toolchain repro are kept as the validated artifact with explicit re-entry
  triggers (LUT kernel, Rust trainer move, fusion goals).

**Updated one-line summary (§5's prediction, scored):** *the framework was small and the data fixed,
and the port to JAX (gymnax-style env + PureJaxRL-style trainer) delivered GPU-resident training in
days, not weeks — 187× end-to-end and swarm-affordable sweeps; PufferLib was unnecessary once the
env was native; and the cutile/cuda-oxide spike produced the predicted shape of answer — a real,
parity-exact kernel that beats every baseline except the one that matters (the JAX env it would
have replaced), so the pipeline stays on JAX with Rust/CUDA readiness banked.*
