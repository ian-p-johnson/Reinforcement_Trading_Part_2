# ARTIFACTS.md — Frozen artifact & schema conventions

**Status:** frozen at Track A (DECISIONS.md P4) so every engine (SB3 today, JAX/Rust later) drops
into the same harness: `train_ppo._passes_consistency_gate`, `_finalize_deployment`,
`final_holdout_eval.py`, `training_diagnostics.py`, `view_results.py`.

## Per-fold directory layout (`models/sliding/fold_k/` or `models/walk_forward/fold_k/`)

| File | Producer | Contents |
|---|---|---|
| `run_info.json` | `train()` | slug, model_path, vecnorm_path, decision_timeframe, sl/tp multipliers, total_timesteps, train_episode_steps, seed, dd_penalty, risk_fraction, spread_price; optional `best_model_path`, `best_model_vecnorm_norm_path`, `promoted_from_fold`, `gate_passed` |
| `<slug>.zip` / `<slug>_vecnorm.pkl` | `train()` | final weights + final VecNormalize snapshot |
| `best_model/best_model.zip` | `_ConsistencyEvalCallback` | best (eligible) checkpoint |
| `best_model/best_model_vecnorm.pkl` | `_ConsistencyEvalCallback` | **the normalisation it was selected under** — always travel together (invariant) |
| `eval_logs/consistency_evals.csv` | `_ConsistencyEvalCallback` | columns: timesteps, train_eval_r, val_r, train_dd_pct, val_dd_pct, q_train, q_val, score, gap, eligible |
| `<slug>_equity_insample_oos.html` | `_post_training_eval` | in-sample → OOS chart (A6: skipped in non-final sliding folds) |
| `fold_result.json` | `_run_sliding_fold` (A4) | the fold's summary row (JSON, dates as ISO strings) |
| `test_equity.csv`, `test_trades.csv` | `_run_sliding_fold` (A4) | fold test-window rollout artifacts for the parent orchestrator |
| `fold_spec.json`, `worker.log` | parallel driver (A4) | worker input spec + its captured stdout |

## Production slot (`models/`)

`run_info.json` (with `promoted_from_fold`, `gate_passed`), `best_model/`, `eval_logs/`,
`NO_DEPLOY.txt` (present iff the consistency gate failed). `final_holdout_eval.py` reads
`run_info.json` and prefers `best_model/best_model.zip` **with** its
`best_model/best_model_vecnorm.pkl` when present.

## Sweep-level artifacts (`models/sliding/`)

| File | Contents |
|---|---|
| `sliding_walk_forward_summary.csv` | one row per fold: fold, train_start/end, test_start/end, val_return_pct, val_profit_factor, test_return_pct, test_sharpe, test_profit_factor, test_win_rate_pct, test_max_dd_pct, test_avg_r, test_n_trades |
| `sliding_oos_equity.csv` | stitched compounded OOS equity (`equity` column) |
| `sliding_walk_forward_summary.csv` for block WF | `train_bars, val_bars, val_start/end, val_*` metrics |

## Gate semantics (unchanged)

`_passes_consistency_gate(summary, ret_col, pf_col, sharpe_col)`: ≥ `min_consistent_folds` folds with
return>0 AND PF> `gate_min_profit_factor`; worst-fold PF ≥ `gate_worst_fold_min_pf`; mean Sharpe > 0.
The gate **only prevents** deployment (`NO_DEPLOY.txt`); the fold's best model is always saved.

## Invariants that any new engine must preserve

1. VecNormalize-equivalent stats **snapshotted with** the checkpoint they were selected under.
2. Fold test windows are stitched in fold order with compounding
   `scaled = equity / initial_equity * running`.
3. `run_info.json` keys above are stable; new engines add keys, never remove/rename.
4. The sealed test is byte-identical across split schemes (tested: `tests/test_splits.py`).
