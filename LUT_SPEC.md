# LUT_SPEC.md — First-touch lookup: specification for the GPU/vectorized port

**Status:** design reference for Track B (JAX) and Track C (cutile/cuda-oxide), owner Track A.
**Why this file exists:** the naive "first-touch table per (bar × bracket)" design sketched in
`SCALING_GPU.md` §1 is **wrong for held positions**. The subtlety is documented here (with the
exact fix) so the vectorized ports do not silently reproduce it. The Python env intentionally does
*not* use a LUT — it uses a vectorized window scan (DECISIONS.md D3/D4), which is exact and fast
enough at single-env speed; the LUT pays off only when thousands of environments are resident.

---

## 1. What must be simulated

Per `env_bracket.py`, a position opened at decision bar *e* has fixed bracket prices:

```
entry = close[e] ± (spread/2 + slippage)          (sign = direction)
sl    = entry - dir · sl_mult · atr[e]
tp    = entry + dir · tp_r   · sl_mult · atr[e]
```

On every subsequent decision step the environment scans the **next** M1 window
``(decision_index[i], decision_index[i+1]]`` and resolves the first touch:

* window is half-open: `searchsorted(start, "right") … searchsorted(end, "right")`,
* within one M1 bar, **SL is checked before TP** (pessimistic rule) → same-bar double touch = SL,
* the first window bar (in time order) that touches either level decides; ties (same bar) → SL,
* a window with no M1 bars leaves the position untouched.

## 2. The subtlety: entry bar ≠ window bar

The bracket prices come from the **entry** bar `e`, but the scan runs against the **current**
window `i` for every step the position is held. So the touch result is a function of
**(e, i, direction, sl_mult, tp_r)** — not of `(e, combo)` alone.

A table indexed only by `(bar, combo)` is correct **only for the first window after entry**.
Reusing it for later windows is a correctness bug: a position held for 40 bars would be resolved
against entry-window price geometry. The naive LUT would silently agree with the reference on
short trades (most trades!) and diverge on long holds — exactly the kind of regression the golden
fixtures cannot catch unless a long-hold scenario exists (it does: `long_tight_reentry`, `gap_3days`).

Table size if built naively as `(entry_bar × window_bar × combo)`: **O(T²) ≈ 137,710² × 24** — not
constructible.

## 3. Exact design: per-window monotone step functions

Observation: for a **fixed window** `w` and a fixed side, the answer "first bar index whose `low`
≤ X" (or "`high` ≥ X") is a monotone step function of the threshold X. Precompute it once per
window per side; the runtime lookup is O(log W) (or O(1) with a bucketed approximation).

### 3.1 Construction (host side, once per dataset)

For window `w` with M1 bars `j = lo … hi-1` (up to 60 for H1):

* **SL side (long)**: thresholds are the bars' `low[j]`; `first_sl(X) = min{ j : low[j] ≤ X }`.
  Build breakpoints by sorting bars by `low` **descending**, keeping a running minimum of `j`;
  each breakpoint stores `(threshold, answer_j)`. Monotone: as X grows, the answer index can only
  move earlier. A sentinel ("no touch") for `X < min(low)`.
* **TP side (long)**: same with `high[j]`, predicate `high[j] ≥ X`.
* **Shorts** swap the sides (SL touches by `high`, TP by `low`) — no extra storage needed.

Storage (per window, per side): ≤ 60 `(f64 threshold, i16 offset)` pairs, padded. For
137,710 windows × 2 sides × 60 entries × 10 B ≈ **165 MB f64 / ~100 MB f32+i16** — fine on a
12 GB GPU; per-fold slices are ~5× smaller. (Exact, no approximation.)

### 3.2 Runtime lookup (per env step, while in trade)

```
sl_X = sl price (from entry bar e)      tp_X = tp price (from entry bar e)
j_sl = first_sl[window i](sl_X)         j_tp = first_tp[window i](tp_X)
if j_sl is not None and (j_tp is None or j_sl <= j_tp):  exit SL at m1_time[j_sl]
elif j_tp is not None:                                   exit TP at m1_time[j_tp]
else:                                                    hold
```

This is exact — including same-bar double touches (ties → SL) and empty windows (both lookups
return None) — and O(log 60) per step.

### 3.3 GPU formulation (Track B/C)

* **JAX:** the breakpoint arrays are static device params; the lookup is a `vmap`-friendly
  `searchsorted` over a row (or, for fixed ≤60-width windows, a padded gather + `argmax` of the
  mask `low_padded ≤ X` with an any() guard — same cost class, simpler to compile). With
  `jax_enable_x64` thresholds are exact; f32 thresholds are safe only if the entry-price
  arithmetic is done in f32 consistently (validate against golden traces).
* **cutile / cuda-oxide:** one thread/tile per env; the step-function rows are read-only global
  tensors; the lookup is a tiny per-thread binary search (SIMT: `__ffs`-style masked scans also
  work). Because the dataset is fixed for the whole run, the breakpoint tensors are loaded once
  and never written.

### 3.4 Alternative (simpler, also exact): padded-window gather

Skip the step function entirely: pad each window to 60 M1 bars with a validity mask, then per step
compute `first_touch = argmax(mask & (low_padded ≤ X))` (guarded by `any`). Cost: 2 masked
scans of 60 elements per step. In JAX this is a `vmap` over `(batch, 60)` — trivially fast on GPU.
In the Python env this is what `env_bracket._simulate_m1_until_next_decision` now does via numpy
(DECISIONS.md D3). **Recommendation: start here; only build the step-function LUT if profiling
shows the scan matters (it should not — the gather is O(60) arithmetic per step).**

## 4. Validation contract (for whoever implements it)

1. Differential test against the pandas reference (`tests/test_intrabar.py` carries a verbatim copy
   of the original per-bar loop as oracle).
2. Must cover: same-bar double touch → SL; empty window; window at the data end; level exactly at
   a bar's high/low (boundary `≤` / `≥` inclusivity); multi-window holds where the entry bar is far
   from the current window (the §2 case).
3. Golden traces (`tests/golden/*.npz`) must match bit-for-bit in f64 mode (or within a documented
   f32 tolerance) for all ten scenarios.

## 5. Summary for the port

| Question | Answer |
|---|---|
| Can the LUT key be `(bar, combo)`? | **No** — only for the entry window. Key must include the current window (§2). |
| Exact, O(1)-ish design? | Per-window monotone step functions (§3.1–3.2). |
| Simpler exact design? | Padded-window masked scan (§3.4) — used in the Python env today. |
| Memory? | ~100–165 MB f32/i16 for the full H1 dataset; trivial on GPU, fine in RAM. |
| Wrong-but-seductive shortcut? | Reusing entry-window touches for later windows. |

**Bottom line:** the market data is fixed and brackets are discrete, so the *whole* simulation step
collapses to gathers + a handful of comparisons — which is what makes this environment such a good
fit for `vmap` (JAX) or a resident CUDA kernel. Just index the LUT by (entry, window, combo) or use
the padded-scan form; do not index it by (bar, combo) alone.
