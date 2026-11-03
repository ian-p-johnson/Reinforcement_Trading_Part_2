# toolchain.md — Track C toolchain verification (C0, SCALING_PLAN.md)

**Machine:** RTX 5070 Ti Laptop (sm_120), driver 590.48.01 (CUDA 13.1), Ubuntu 24.04, Python 3.12.
**Result:** all three surfaces PASS — cuTile Python, cutile-rs (tile), cuda-oxide (SIMT).
The spike proceeds with **both formulations available**; the env kernel is implemented in
cuTile Python (see `DECISIONS.md` Track C rows for the seam rationale).

## 0. CUDA toolkit sources on this machine

| Source | Version | Notes |
|---|---|---|
| System `/usr/local/cuda-13.1` | 13.1.115 (nvcc), tileiras 13.1 | pre-existing; Tile IR **13.1** — too old for sm_120 tile work and for `load_advanced_indexing` |
| pip `cuda-toolkit` (+ `nvidia-cuda-nvcc/nvvm/crt/tileiras`) | **13.4.92** | installed into the project `.venv` under `site-packages/nvidia/cu13/` — full toolkit (bin/include/lib/nvvm), tileiras emits **Tile IR 13.4** |
| pip `cuda-tile[tileiras]` | cuda-tile 1.7.0 | cuTile Python DSL + JAX FFI |

Version rule learned the hard way: `nvidia-cuda-tileiras` must match `nvidia-cuda-nvcc`/
`nvidia-nvvm` major.minor, and **sm_120 needs Tile IR ≥ 13.2** while
`ct.load_advanced_indexing` needs **tileiras ≥ 13.3** (gather/scatter = Tile IR 13.3
feature, same matrix as cutile-rs). The first install (`cuda-tile[tileiras]` alone) pulled
tileiras/nvcc **13.2.86**, which passes vector-add but fails on the gather op. Fix used:

```
pip install "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3"     # landed on 13.4.92
```

Side effects (verified benign):
* it **downgraded** `nvidia-cuda-nvcc` 13.4.92 → 13.2.86 on the first install (then upgraded
  back to 13.4.92); jax-cuda13-plugin and torch both still work (checked matmuls after each).
* `torch 2.15.0` declares `cuda-toolkit[cublas,…]==13.2.2` — pip now warns about a version
  conflict with `cuda-toolkit 13.4.3`. The meta-package version is what conflicts; torch's
  actual CUDA libs are separate `nvidia-*` wheels and torch CUDA tests pass. Revisit if torch
  upgrades break.

`CUDA_TOOLKIT_PATH` used for the Rust surfaces:
`/media/ianj/New Volume/dev/Reinforcement_Trading_Part_2/.venv/lib/python3.12/site-packages/nvidia/cu13`

## 1. cuTile Python — PASS

```
pip install cuda-tile[tileiras] cupy-cuda13x
```

* f64 vector-add on sm_120: bit-exact (max err 0.0), JIT cache works on relaunch.
* API notes for cuda-tile **1.7.0** (docs online show older call conventions):
  * kernels are launched with `ct.launch(stream, grid, kernel, args)` (NOT `kernel(grid, …)`);
  * **kernels must live in real `.py` files** — the compiler reads the source AST
    (`inspect.findsource`), so stdin/REPL definitions fail with `OSError: could not get source code`;
  * JAX seam: `cuda.tile.jax.cutile_call(grid, kernel, args)` + `OutputPlaceholder`/
    `InputOutput` — present and importable (`HAS_JAX=True` with jax installed).
* Language gotchas hit (alpha surface):
  * `x.astype(ct.int64)` on an int32 tile → tileiras verification error
    (`cuda_tile.trunci op failed to verify that all of {from, to} have same shape`) —
    **avoid int→int casts**; keep index tiles int32 end-to-end.
  * `hasattr()` in kernel code → `UnsupportedCallError` (no reflection in device code) — fair.
  * `ct.reduce(x, axis, func=ct.minimum, identity=…)` works for per-row min-index scans
    (used for the first-touch lookup); dynamic per-lane extract is expressible as a
    one-hot min-reduce over the loaded row tile.

## 2. cutile-rs (Rust tile surface) — PASS

Per the crate docs (cutile 0.4.0, crates.io; repo now at `NVIDIA/cuda-rust/cutile-rs`):

```bash
cargo new cutile_demo && cd cutile_demo
cargo add cutile
# .cargo/config.toml:
#   [env]
#   CUDA_TOOLKIT_PATH = { value = "<venv>/site-packages/nvidia/cu13", relative = false }
cargo run     # README quick-start vecadd
```

* Result: `cutile-rs vecadd on this GPU: OK (1024 f32 elements)` — Tile IR JIT through
  tileiras, sm_120 accepted (matrix: sm_120 needs Tile IR ≥ 13.2 ✓).
* **Rust 1.95 nightly works** (crate asks for stable ≥ 1.89).
* Gotcha: host `cuda-bindings` builds with bindgen → `fatal error: 'stddef.h' file not found`
  (the docs' predicted trap, no clang on this box). Workaround, no sudo needed:
  `export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"`.
* Capability notes from the compatibility matrix (relevant to the env kernel):
  **gather/scatter views, strided views and allocation need Tile IR 13.3** — same constraint
  as `ct.load_advanced_indexing` above. The env kernel's window gather would require the
  13.3+ toolchain either way.

## 3. cuda-oxide (Rust SIMT surface) — PASS

Per `NVIDIA/cuda-rust` docs (the repo moved from `NVlabs/cuda-oxide`):

```bash
git clone --depth 1 https://github.com/NVIDIA/cuda-rust.git
rustup toolchain install 1.99.1 --profile minimal \
    --component rust-src --component rustc-dev --component llvm-tools   # ~2 min
cd cuda-rust/cuda-oxide
export CUDA_TOOLKIT_PATH="<venv>/site-packages/nvidia/cu13"
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
cargo oxide doctor
cargo oxide run vecadd
```

* `cargo oxide run vecadd` → `✓ SUCCESS: All 1024 elements correct!` on sm_120, driver 590
  (the PTX path works here — the plan's PTX-vs-driver-580 trap did not reproduce on 590).
* `cargo oxide doctor`: all checks green except `clang resource dir` (same bindgen family —
  the `BINDGEN_EXTRA_CLANG_ARGS` workaround suffices; `sudo apt install clang-21` is the
  documented fix if the workaround ever stops working).
* Toolchain pin is `1.99.1` via `rust-toolchain.toml` (rustc 1.99.1, llvm-tools LLVM 23 —
  `llc` resolved from `rustup component add llvm-tools`, no system LLVM needed).
* Codegen backend builds automatically on first `cargo oxide` invocation (~3 min cold).

## 4. Repro checklist (fresh machine)

1. `pip install cuda-tile[tileiras] cupy-cuda13x "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3"`
2. Rust: stable ≥1.89 for cutile-rs; `1.99.1` + `rust-src rustc-dev llvm-tools` for cuda-oxide.
3. `export CUDA_TOOLKIT_PATH=<venv>/site-packages/nvidia/cu13`
4. `export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"`
   (or install `clang-21`).
5. Smoke: `python tests/…` (cuTile parity suite), `cargo run` in a `cargo add cutile` project,
   `cargo oxide run vecadd`.

## 5. Kernel-level findings (cutile_env.py implementation, cuda-tile 1.7.0)

Encountered while implementing the env-step kernel — each one cost a bisection,
so they are worth more than the install notes:

| Finding | Symptom | Workaround used |
|---|---|---|
| **Kernel scalar ABI is f32** | python/`np.float64` scalars passed to `ct.launch` arrive as float32 (e.g. `10000.0*0.005` → `49.99999888`); `: ct.float64` annotations do not help | route all scalars through **f64/i32 buffers** (`scalar_arr [8]`, `int_arr [2]`) + `ct.load`/`ct.extract` — exact |
| **`x.astype(ct.int64)` on an int32 tile** | tileiras verification error `cuda_tile.trunci … {from,to} have same shape` | keep index tiles int32 end-to-end; cast i32→i64 via `where(i==0, zero_i64, x_i64)`-style selects or not at all |
| **i64 min-reduce with a large identity** (`ct.reduce(where(onehot, ts_row, INT64_MAX), axis=1, min)`) | *silent miscompile*: neighbouring SSA values get corrupted (observed: entry prices recomputed from a different row's `close`) | replaced with `ct.gather(ts_pad, (row, rel))` for the dynamic extract; the int32 min-reduces are fine |
| **`ct.load_advanced_indexing` needs tileiras ≥ 13.3** | `TileUnsupportedFeatureError` on 13.2 (gather/scatter = Tile IR 13.3) | `pip install "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3"` |
| **1-D gathers use `ct.gather(array, (idx,))`** | `load_advanced_indexing` rejects rank-1 arrays | `ct.gather` — correct with per-lane varying indices (verified) |
| **Broadcast footgun** | `low_tile[TILE,W] <= sl_vec[TILE]` broadcasts the vector as `(1, W)` (per-column!), silently producing wrong touches | reshape per-lane vectors to `(TILE, 1)` whenever mixing with the window plane |
| **Tiles are not indexable** (`sm[0]`) | `TileTypeError … use extract()` | `ct.extract(tile, (k,), (1,))` for constant indices |
| **Kernels must be defined in real `.py` files** | `OSError: could not get source code` from stdin/REPL | the compiler reads the source AST (`inspect.findsource`) |
| **f64 exactness vs XLA** | 1-ulp (≈2e-13) differences vs `jax_env.step` on some chains (e.g. `tp = entry + d*tp_r*sl_dist`) — compiler association/FMA differs | parity tolerance `atol=1e-9` on f64 per D32; ints/bools/obs are exact |
| Nested `def` in kernel code | works (1.7.0) | — |

cuda-oxide/cutile-rs gotchas are in §2/§3 above (bindgen `stddef.h`, tileiras mixing).
