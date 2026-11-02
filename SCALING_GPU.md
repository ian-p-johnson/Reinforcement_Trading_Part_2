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
