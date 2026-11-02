"""JAX throughput benchmarks (Track B5, SCALING_PLAN.md B5).

Measures and records:

  * env-only steps/s — jitted ``vmap`` step at 1k/4k/16k/64k envs,
  * end-to-end train steps/s (rollout + GAE + PPO update),
  * compile time per shape (rollout / postprocess / update / eval),
  * host↔device volume per rollout (trajectory nbytes),
  * VRAM residency around rollout+update.

JIT hygiene (B5): one jit closure per config, all varying values passed as
*arguments* (``jax_ppo`` already does this — env_params/lr/rng are call args),
``jax.clear_caches()`` between discrete experiments, static shapes within a run.

Timing policy: the recorded numbers must come from a quiet machine — run the
full suite there (``python jax_benchmarks.py all``).  ``--quick`` exists only to
verify the harness wiring and its output must never be quoted as a result.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

import jax
import jax.numpy as jnp

import jax_env
import jax_ppo

RESULTS: list[tuple[str, str, str]] = []


def record(group: str, name: str, detail: str) -> None:
    RESULTS.append((group, name, detail))
    print(f"[{group:5s}] {name:52s} {detail}", flush=True)


# ── data for benchmark shapes ───────────────────────────────────────────────


def _bench_params(data: str, episode_steps: int | None = None) -> jax_env.EnvParams:
    """EnvParams at production-like shapes: real fold slice (default) or the
    hermetic synthetic scenario data.  Both give W=60 windows."""
    if data == "real":
        import train_ppo as tp
        from data_loader import make_sliding_folds
        from config import CFG
        m1, feat, feature_cols = tp._load_decision_features()
        folds = make_sliding_folds(
            feat, train_years=CFG.sliding_train_years,
            val_months=CFG.sliding_val_months, test_months=CFG.sliding_test_months,
            step_months=CFG.sliding_step_months, embargo_bars=CFG.split_embargo_bars)
        tr, _va, _te = folds[0]
        m1_sl = tp._slice_m1_for_decision_window(m1, tr)
        return jax_ppo.build_env_params_cfg(tr, m1_sl, feature_cols, episode_steps)
    from scenarios import build_frame, make_m1, m1_slice_for
    m1 = make_m1(n_days=30, seed=7)
    feat, cols = build_frame(m1)
    return jax_ppo.build_env_params_cfg(
        feat, m1_slice_for(m1, feat), cols, episode_steps)


def _init_train_state(rng, params_env, num_envs, obs_dim):
    norm = jax_ppo.init_norm_state(obs_dim)
    reset_rngs = jax.random.split(rng, num_envs)
    obs_raw, env_state = jax.vmap(jax_env.reset, in_axes=(0, None))(reset_rngs, params_env)
    norm = jax_ppo.update_obs_rms(norm, obs_raw)
    obs_norm = jax_ppo.normalize_obs(obs_raw, norm)
    net_params = jax_ppo.init_params(rng, obs_dim)
    opt = jax_ppo.make_optimizer(6e-5, 1e-5)
    return jax_ppo.TrainState(
        params=net_params, opt_state=opt.init(net_params), norm=norm,
        returns=jnp.zeros((num_envs,), dtype=jnp.float64),
        env_state=env_state, obs_norm=obs_norm, rng=rng,
        update_idx=jnp.asarray(0, dtype=jnp.int32))


# ── groups ──────────────────────────────────────────────────────────────────


def bench_env(data: str = "real", quick: bool = False) -> None:
    """Jitted vmap step throughput at several env widths."""
    widths = (8, 16) if quick else (1_024, 4_096, 16_384, 65_536)
    scan_steps = 2 if quick else 64
    params = _bench_params(data)
    obs_dim = params.feat.shape[1] + 6

    for n in widths:
        jax.clear_caches()
        state = jax.vmap(lambda i: jax_env.init_state(params, i))(
            jnp.zeros((n,), dtype=jnp.int32))
        actions = jnp.zeros((n, 3), dtype=jnp.int32)
        rngs = jax.random.split(jax.random.PRNGKey(0), n)

        def scan_body(carry, _):
            state, rngs = carry
            splits = jax.vmap(jax.random.split)(rngs)      # [n, 2, 2]
            rngs_new, reset_rngs = splits[:, 0], splits[:, 1]
            obs, state, rew, term, trunc, info, _f, _r = jax.vmap(
                jax_env.step_autoreset, in_axes=(0, 0, 0, None))(
                reset_rngs, state, actions, params)
            return (state, rngs_new), rew.mean()

        fn = jax.jit(lambda s, r: jax.lax.scan(scan_body, (s, r), None, length=scan_steps))
        t0 = time.perf_counter()
        (state, _), _ = fn(state, rngs)          # compile
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        (state, _), out = fn(state, rngs)        # timed
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        sps = n * scan_steps / dt
        record("env", f"vmap step @ {n:,} envs ({params.feat.shape[0]:,} bars)",
               f"{sps:,.0f} steps/s  (compile {t_compile:.1f}s)")


def bench_train(data: str = "real", quick: bool = False) -> None:
    """End-to-end train steps/s (rollout + GAE + update) at CFG geometry."""
    from config import CFG
    n_envs = 8 if quick else CFG.jax_num_envs
    n_steps = 4 if quick else CFG.jax_num_steps
    n_mb = 2 if quick else CFG.jax_num_minibatches
    updates = 2 if quick else 10

    params = _bench_params(data, episode_steps=2048)
    obs_dim = params.feat.shape[1] + 6
    jax.clear_caches()
    state = _init_train_state(jax.random.PRNGKey(0), params, n_envs, obs_dim)

    rollout_fn = jax_ppo.make_rollout_fn(n_envs, n_steps, 0.99)
    post_fn = jax_ppo.make_trajectory_postprocess(n_envs, n_steps, 0.99, 0.95)
    update_fn = jax_ppo.make_update_fn(n_mb, CFG.ppo_n_epochs, CFG.ppo_clip_range,
                                       CFG.ppo_ent_coef, 0.5, CFG.ppo_target_kl,
                                       CFG.ppo_weight_decay)

    def one_update(state, env_params, lr):
        state, traj = rollout_fn(state, env_params)
        traj = post_fn(traj)
        p, o, m = update_fn(state.params, state.opt_state, traj, state.rng,
                            jnp.asarray(lr, dtype=jnp.float32))
        return state._replace(params=p, opt_state=o), m

    fn = jax.jit(one_update)
    t0 = time.perf_counter()
    state, _ = fn(state, params, jnp.float32(6e-5))       # compile
    t_compile = time.perf_counter() - t0

    buffer = n_envs * n_steps
    t0 = time.perf_counter()
    for k in range(updates):
        state, m = fn(state, params, jnp.float32(6e-5 * (1 - k / updates)))
    jax.block_until_ready(m)
    dt = time.perf_counter() - t0
    sps = buffer * updates / dt
    record("train", f"end-to-end {n_envs}x{n_steps} (buffer {buffer:,})",
           f"{sps:,.0f} steps/s  (compile {t_compile:.1f}s; "
           f"{updates * buffer:,} steps in {dt:.1f}s)")


def bench_compile(data: str = "real", quick: bool = False) -> None:
    """First-call (compile) time per jitted component and shape."""
    from config import CFG
    shapes = [(8, 4, 2)] if quick else [
        (4, 2048, 32), (CFG.jax_num_envs, CFG.jax_num_steps, CFG.jax_num_minibatches)]
    for n_envs, n_steps, n_mb in shapes:
        params = _bench_params(data, episode_steps=2048)
        obs_dim = params.feat.shape[1] + 6
        state = _init_train_state(jax.random.PRNGKey(0), params, n_envs, obs_dim)
        for name, build in (
            ("rollout", lambda: jax_ppo.make_rollout_fn(n_envs, n_steps, 0.99)),
            ("gae_post", lambda: jax_ppo.make_trajectory_postprocess(
                n_envs, n_steps, 0.99, 0.95)),
            ("update", lambda: jax_ppo.make_update_fn(
                n_mb, 5, 0.1, 0.03, 0.5, None, 1e-5)),
        ):
            jax.clear_caches()
            fn = build()
            t0 = time.perf_counter()
            if name == "rollout":
                state2, traj = fn(state, params)
                jax.block_until_ready(traj)
            elif name == "gae_post":
                _, traj = jax_ppo.make_rollout_fn(n_envs, n_steps, 0.99)(state, params)
                jax.clear_caches()
                fn = build()
                t0 = time.perf_counter()
                out = fn(traj)
                jax.block_until_ready(out)
            else:
                _, traj = jax_ppo.make_rollout_fn(n_envs, n_steps, 0.99)(state, params)
                traj = jax_ppo.make_trajectory_postprocess(n_envs, n_steps, 0.99, 0.95)(traj)
                jax.clear_caches()
                fn = build()
                t0 = time.perf_counter()
                p, o, m = fn(state.params, state.opt_state, traj, state.rng,
                             jnp.float32(6e-5))
                jax.block_until_ready(m)
            dt = time.perf_counter() - t0
            record("comp", f"{name} @ {n_envs}x{n_steps}/mb{n_mb}", f"{dt:.2f}s")


def bench_transfer(data: str = "real", quick: bool = False) -> None:
    """Host↔device volume of one rollout trajectory + measured pull cost."""
    from config import CFG
    n_envs = 8 if quick else CFG.jax_num_envs
    n_steps = 4 if quick else CFG.jax_num_steps
    params = _bench_params(data, episode_steps=2048)
    obs_dim = params.feat.shape[1] + 6
    state = _init_train_state(jax.random.PRNGKey(0), params, n_envs, obs_dim)
    rollout_fn = jax_ppo.make_rollout_fn(n_envs, n_steps, 0.99)
    state, traj = rollout_fn(state, params)

    nbytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(traj))
    record("xfer", f"trajectory per rollout ({n_envs}x{n_steps})",
           f"{nbytes / 1e6:.2f} MB device→host if pulled")

    host = jax.tree_util.tree_map(np.asarray, traj)
    jax.block_until_ready(host)
    t0 = time.perf_counter()
    host = jax.tree_util.tree_map(np.asarray, traj)
    jax.block_until_ready(host)
    dt = time.perf_counter() - t0
    record("xfer", "device→host pull of one trajectory", f"{dt * 1e3:.1f} ms")


def bench_vram(data: str = "real", quick: bool = False) -> None:
    """VRAM residency around rollout + update at CFG geometry."""
    from config import CFG
    n_envs = 8 if quick else CFG.jax_num_envs
    n_steps = 4 if quick else CFG.jax_num_steps
    n_mb = 2 if quick else CFG.jax_num_minibatches
    params = _bench_params(data, episode_steps=2048)
    obs_dim = params.feat.shape[1] + 6
    jax.clear_caches()
    state = _init_train_state(jax.random.PRNGKey(0), params, n_envs, obs_dim)
    rollout_fn = jax_ppo.make_rollout_fn(n_envs, n_steps, 0.99)
    update_fn = jax_ppo.make_update_fn(n_mb, 5, 0.1, 0.03, 0.5, None, 1e-5)
    post_fn = jax_ppo.make_trajectory_postprocess(n_envs, n_steps, 0.99, 0.95)

    dev = jax.devices()[0]

    def stats():
        try:
            s = dev.memory_stats()
            return s.get("bytes_in_use", 0), s.get("peak_bytes_in_use", 0)
        except Exception:
            return -1, -1

    before = stats()
    state, traj = rollout_fn(state, params)
    traj = post_fn(traj)
    p, o, m = update_fn(state.params, state.opt_state, traj, state.rng,
                        jnp.float32(6e-5))
    jax.block_until_ready(m)
    after = stats()
    record("vram", f"rollout+update @ {n_envs}x{n_steps}",
           f"in_use {after[0] / 1e6:,.0f} MB (peak {after[1] / 1e6:,.0f} MB; "
           f"was {before[0] / 1e6:,.0f} MB)")


GROUPS = {
    "env": bench_env,
    "train": bench_train,
    "comp": bench_compile,
    "xfer": bench_transfer,
    "vram": bench_vram,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("groups", nargs="*", default=["env"],
                    choices=sorted(GROUPS) + ["all"])
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--quick", action="store_true",
                    help="harness sanity at tiny sizes — NOT quotable numbers")
    args = ap.parse_args()
    groups = list(GROUPS) if "all" in args.groups else args.groups
    if args.quick:
        print("!! --quick: harness sanity only — do not quote these numbers")
    for g in groups:
        print(f"\n=== {g} ===", flush=True)
        GROUPS[g](data=args.data, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
