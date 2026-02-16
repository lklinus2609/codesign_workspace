#!/usr/bin/env python3
"""
PGHC Co-Design for G1 Humanoid — JAX/MJX Implementation

Main entry point implementing Algorithm 1 from the thesis:
    for outer_iter in range(max_outer_iters):
        1. apply_theta -> update MJX model body quaternions
        2. PPO inner loop training until convergence
        3. Collect actions (frozen policy, deterministic)
        4. BPTT gradient via jax.grad through mjx.step
        5. Adam update theta, clip to +/-30 deg
        6. Log metrics, save checkpoint
        7. Check outer convergence

Key advantage: both inner and outer loops use identical MJX physics,
eliminating the solver mismatch that caused training failures.

Run:
    python pghc_jax.py --num-envs 4096 --wandb
"""

import argparse
import os
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import mujoco
import numpy as np

from g1_model import load_g1_model, NUM_DESIGN_PARAMS, SYMMETRIC_PAIRS
from g1_morphology import apply_theta
from g1_env_jax import make_env_fns, EnvConfig, EnvState
from g1_bptt_jax import make_bptt_fns, compute_bptt_gradient
from ppo_jax import (
    init_networks, sample_action, deterministic_action,
    collect_actions_deterministic, compute_gae, ppo_loss,
    make_ppo_train_fns, PPOConfig, Transition, StabilityGate,
)
from utils import AdamOptimizer, RunningMeanStd, PGHCLogger


# ---------------------------------------------------------------------------
# Video recording (MuJoCo CPU renderer -> wandb)
# ---------------------------------------------------------------------------

def record_video(mj_model, env_fns, policy_apply, policy_params, obs_rms,
                 rng, num_steps=200, width=480, height=360):
    """Record a short video of the current policy using MuJoCo CPU renderer.

    Args:
        mj_model: mujoco.MjModel (CPU)
        env_fns: environment functions dict
        policy_apply: policy network apply function
        policy_params: frozen policy parameters
        obs_rms: observation normalizer
        rng: PRNGKey
        num_steps: number of simulation steps to record
        width, height: video resolution

    Returns:
        frames: (T, H, W, 3) uint8 numpy array, or None if rendering fails
    """
    try:
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
    except Exception:
        return None

    mj_data = mujoco.MjData(mj_model)

    # Reset a single env and get initial qpos
    rng, rng_reset = jax.random.split(rng)
    env_state, obs = env_fns["reset_single"](rng_reset)

    # Copy MJX state to CPU
    mj_data.qpos[:] = np.asarray(env_state.mjx_data.qpos)
    mj_data.qvel[:] = np.asarray(env_state.mjx_data.qvel)
    mujoco.mj_forward(mj_model, mj_data)

    # Camera setup: follow the robot from the side
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.0
    cam.elevation = -20.0
    cam.azimuth = 90.0

    frames = []
    # Render every Nth step (~50 fps at dt=0.002)
    render_every = max(1, int(0.02 / mj_model.opt.timestep))

    for step_i in range(num_steps):
        # Get action from policy
        obs_np = np.asarray(obs)[None]  # (1, obs_dim)
        obs_norm = jnp.array(obs_rms.normalize(obs_np))[0]
        action = deterministic_action(policy_apply, policy_params, obs_norm)

        # Step environment (JAX)
        env_state, obs, _, done, _ = env_fns["step_single"](env_state, action)

        # Copy to CPU renderer
        mj_data.qpos[:] = np.asarray(env_state.mjx_data.qpos)
        mj_data.qvel[:] = np.asarray(env_state.mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        if step_i % render_every == 0:
            # Track the robot's root position
            cam.lookat[:] = mj_data.qpos[:3]
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render().copy())

    renderer.close()

    if not frames:
        return None
    return np.stack(frames)  # (T, H, W, 3)


# ---------------------------------------------------------------------------
# Inner loop: PPO training
# ---------------------------------------------------------------------------

def run_inner_loop(env_fns, policy_params, value_params, opt_state,
                   ppo_train_fns, obs_rms, rng,
                   ppo_cfg, gate, max_samples=200_000_000,
                   log_every=10, use_wandb=False, logger=None,
                   mj_model=None, video_every=100,
                   env_sharding=None):
    """Train PPO until convergence or sample cap.

    Args:
        env_fns: dict from make_env_fns()
        policy_params, value_params: network parameters
        opt_state: optimizer state
        ppo_train_fns: dict from make_ppo_train_fns()
        obs_rms: RunningMeanStd for observation normalization
        rng: JAX PRNGKey
        ppo_cfg: PPOConfig
        gate: StabilityGate for convergence detection
        max_samples: maximum samples before stopping
        log_every: print metrics every N rollouts
        use_wandb: whether to log to wandb
        logger: PGHCLogger instance
        mj_model: mujoco.MjModel for video rendering (optional)
        video_every: record video every N rollouts (0 to disable)
        env_sharding: NamedSharding for distributing envs across GPUs

    Returns:
        policy_params, value_params, opt_state, obs_rms, metrics, converged
    """
    policy_apply = ppo_train_fns["policy_apply"]
    value_apply = ppo_train_fns["value_apply"]
    update_fn = ppo_train_fns["update"]

    num_envs = ppo_cfg.num_envs
    horizon = ppo_cfg.horizon
    obs_dim = env_fns["obs_dim"]
    act_dim = env_fns["act_dim"]

    # Reset environments
    rng, rng_reset = jax.random.split(rng)
    rng_envs = jax.random.split(rng_reset, num_envs)
    # Shard RNG keys across devices before reset (triggers parallel reset)
    if env_sharding is not None:
        rng_envs = jax.device_put(rng_envs, env_sharding)
    env_states, obs_batch = env_fns["reset"](rng_envs)

    gate.reset()
    total_samples = 0
    rollout_count = 0
    reward_accum = []
    start_time = time.time()

    print(f"    [Inner] Starting PPO training ({num_envs} envs, "
          f"horizon={horizon})")

    while total_samples < max_samples:
        # --- Collect rollout ---
        transitions_list = []
        rng, rng_rollout = jax.random.split(rng)

        for t in range(horizon):
            # Normalize observations
            obs_np = np.asarray(obs_batch)
            obs_rms.update(obs_np)
            obs_norm = jnp.array(obs_rms.normalize(obs_np))

            # Sample actions
            rng, rng_act = jax.random.split(rng)
            rng_acts = jax.random.split(rng_act, num_envs)
            if env_sharding is not None:
                rng_acts = jax.device_put(rng_acts, env_sharding)
                obs_norm = jax.device_put(obs_norm, env_sharding)
            actions, log_probs = jax.vmap(
                sample_action, in_axes=(None, None, 0, 0)
            )(policy_apply, policy_params, obs_norm, rng_acts)

            # Value predictions
            values = jax.vmap(value_apply, in_axes=(None, 0))(
                value_params, obs_norm)

            # Step environments
            env_states, next_obs, rewards, dones, infos = env_fns["step"](
                env_states, actions)

            transitions_list.append(Transition(
                obs=obs_norm,
                action=actions,
                reward=rewards,
                done=dones.astype(jnp.float32),
                log_prob=log_probs,
                value=values,
            ))

            obs_batch = next_obs

        # Stack transitions: (H, N, ...)
        transitions = jax.tree.map(
            lambda *xs: jnp.stack(xs), *transitions_list
        )

        # Bootstrap value
        obs_np = np.asarray(obs_batch)
        obs_norm = jnp.array(obs_rms.normalize(obs_np))
        last_value = jax.vmap(value_apply, in_axes=(None, 0))(
            value_params, obs_norm)

        # --- PPO update ---
        rng, rng_update = jax.random.split(rng)
        policy_params, value_params, opt_state, ppo_info = update_fn(
            policy_params, value_params, opt_state,
            transitions, last_value, rng_update,
        )

        total_samples += num_envs * horizon
        rollout_count += 1

        # Track rewards
        mean_reward = float(jnp.mean(transitions.reward))
        reward_accum.append(mean_reward)

        # Convergence check
        gate.update(mean_reward)
        gate.tick(1)

        if rollout_count % log_every == 0:
            elapsed = time.time() - start_time
            print(f"      Rollout {rollout_count}: "
                  f"mean_rew={mean_reward:.4f}, "
                  f"samples={total_samples:,}, "
                  f"elapsed={elapsed:.0f}s, "
                  f"policy_loss={float(ppo_info['policy_loss']):.4f}, "
                  f"value_loss={float(ppo_info['value_loss']):.4f}")

            if logger and use_wandb:
                logger.log({
                    "inner/mean_reward": mean_reward,
                    "inner/policy_loss": float(ppo_info["policy_loss"]),
                    "inner/value_loss": float(ppo_info["value_loss"]),
                    "inner/entropy": float(ppo_info["entropy"]),
                    "inner/approx_kl": float(ppo_info["approx_kl"]),
                    "inner/total_samples": total_samples,
                })

        # Record video to wandb
        if (use_wandb and mj_model is not None and video_every > 0
                and rollout_count % video_every == 0):
            try:
                import wandb
                rng, rng_vid = jax.random.split(rng)
                frames = record_video(
                    mj_model, env_fns, policy_apply, policy_params,
                    obs_rms, rng_vid, num_steps=500)
                if frames is not None:
                    # wandb expects (T, C, H, W) for Video
                    vid = np.transpose(frames, (0, 3, 1, 2))
                    logger.log({
                        "inner/video": wandb.Video(vid, fps=50,
                                                   format="mp4"),
                    })
                    print(f"      [Video] Logged {frames.shape[0]} frames "
                          f"to wandb")
            except Exception as e:
                print(f"      [Video] Failed: {e}")

        if gate.is_converged():
            print(f"    [Inner] CONVERGED at rollout {rollout_count} "
                  f"(mean_rew={mean_reward:.4f})")
            break

    converged = gate.is_converged()
    if not converged:
        print(f"    [Inner] Sample cap reached ({total_samples:,})")

    metrics = {
        "total_samples": total_samples,
        "rollout_count": rollout_count,
        "final_mean_reward": reward_accum[-1] if reward_accum else 0.0,
        "converged": converged,
        "elapsed_s": time.time() - start_time,
    }

    return (policy_params, value_params, opt_state, obs_rms,
            metrics, converged)


# ---------------------------------------------------------------------------
# Action collection for BPTT
# ---------------------------------------------------------------------------

def collect_eval_actions(env_fns, policy_apply, policy_params, obs_rms,
                         num_eval_envs, eval_horizon, rng,
                         env_sharding=None):
    """Collect deterministic actions from frozen policy for BPTT.

    Args:
        env_fns: environment functions dict
        policy_apply: policy network apply function
        policy_params: frozen policy parameters
        obs_rms: observation normalizer
        num_eval_envs: number of evaluation environments
        eval_horizon: number of steps to collect
        rng: PRNGKey
        env_sharding: NamedSharding for multi-GPU distribution

    Returns:
        actions: (eval_horizon, num_eval_envs, act_dim) action array
        init_qpos: (num_eval_envs, nq) initial joint positions
    """
    rng, rng_reset = jax.random.split(rng)
    rng_envs = jax.random.split(rng_reset, num_eval_envs)
    if env_sharding is not None:
        rng_envs = jax.device_put(rng_envs, env_sharding)
    env_states, obs_batch = env_fns["reset"](rng_envs)

    # Save initial qpos for BPTT
    init_qpos = env_states.mjx_data.qpos  # (num_eval_envs, nq)

    actions_list = []
    for step in range(eval_horizon):
        obs_np = np.asarray(obs_batch)
        obs_norm = jnp.array(obs_rms.normalize(obs_np))

        # Deterministic actions
        actions = collect_actions_deterministic(policy_apply, policy_params,
                                                obs_norm)
        actions_list.append(actions)

        # Step environments (continue collecting even if some reset)
        env_states, obs_batch, _, _, _ = env_fns["step"](env_states, actions)

    actions = jnp.stack(actions_list)  # (H, N, act_dim)
    # Transpose to (N, H, act_dim) for batched BPTT
    actions = jnp.transpose(actions, (1, 0, 2))

    return actions, init_qpos


# ---------------------------------------------------------------------------
# Main PGHC loop
# ---------------------------------------------------------------------------

def main(args):
    print("=" * 70)
    print("PGHC Co-Design for G1 Humanoid (JAX/MJX)")
    print("=" * 70)

    # Setup output directory
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    use_wandb = args.wandb
    logger = PGHCLogger(out_dir, use_wandb=use_wandb)
    if use_wandb:
        logger.init_wandb(
            project="pghc-codesign-jax",
            name=f"g1-mjx-{args.num_envs}env",
            config=vars(args),
        )

    # JAX device info + multi-GPU sharding
    devices = jax.devices()
    num_devices = len(devices)
    print(f"  JAX devices: {devices}")
    print(f"  JAX backend: {jax.default_backend()}")

    # Create mesh for data-parallel sharding across all GPUs
    mesh = Mesh(np.array(devices), axis_names=('d',))
    # Shard batch dimension across devices
    env_sharding = NamedSharding(mesh, P('d'))
    # Replicate (no sharding) for params, scalars
    replicated = NamedSharding(mesh, P())

    # Ensure num_envs is divisible by num_devices
    if args.num_envs % num_devices != 0:
        old = args.num_envs
        args.num_envs = (args.num_envs // num_devices) * num_devices
        print(f"  [Warning] Adjusted num_envs {old} -> {args.num_envs} "
              f"(divisible by {num_devices} devices)")
    if args.num_eval_envs % num_devices != 0:
        old = args.num_eval_envs
        args.num_eval_envs = (args.num_eval_envs // num_devices) * num_devices
        print(f"  [Warning] Adjusted num_eval_envs {old} -> {args.num_eval_envs} "
              f"(divisible by {num_devices} devices)")

    print(f"  Multi-GPU: {num_devices} devices, "
          f"{args.num_envs // num_devices} envs/device")

    # =====================================================================
    # 1. Load model
    # =====================================================================
    print("\n[1/4] Loading G1 model...")
    mj_model, mjx_model, metadata = load_g1_model()

    obs_dim = 3 + 3 + 3 + 3 * (metadata["nv"] - 6) + 2
    act_dim = metadata["nv"] - 6  # actuated DOFs

    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    # =====================================================================
    # 2. Initialize networks and optimizer
    # =====================================================================
    print("\n[2/4] Initializing networks...")
    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(rng)

    policy_params, value_params, policy_apply, value_apply = init_networks(
        rng_init, obs_dim, act_dim, hidden_dims=(256, 128, 128))

    ppo_cfg = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=0.008,
        value_coeff=1.0,
        max_grad_norm=1.0,
        lr=args.ppo_lr,
        n_epochs=5,
        num_minibatches=4,
        horizon=args.horizon,
        num_envs=args.num_envs,
        desired_kl=0.01,
    )

    ppo_train_fns = make_ppo_train_fns(policy_apply, value_apply, ppo_cfg)
    ppo_train_fns["policy_apply"] = policy_apply
    ppo_train_fns["value_apply"] = value_apply
    opt_state = ppo_train_fns["init_optimizer"](policy_params, value_params)

    obs_rms = RunningMeanStd(obs_dim)

    # =====================================================================
    # 3. Initialize design parameters
    # =====================================================================
    print("\n[3/4] Initializing design parameters...")
    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    design_optimizer = AdamOptimizer(NUM_DESIGN_PARAMS, lr=args.design_lr)
    theta_bounds = (-0.5236, 0.5236)  # +/-30 deg

    param_names = [
        f"theta_{i}_{SYMMETRIC_PAIRS[i][0].replace('_link', '')}"
        for i in range(NUM_DESIGN_PARAMS)
    ]

    # =====================================================================
    # 4. PGHC outer loop
    # =====================================================================
    print("\n[4/4] Starting PGHC outer loop...\n")

    convergence_gate = StabilityGate(
        rel_threshold=args.plateau_threshold,
        min_iters=args.min_inner_iters,
        stable_iters_required=args.stable_iters_required,
        window=5,
    )

    history = {
        "theta": [theta.copy()],
        "forward_dist": [],
        "cot": [],
        "gradients": [],
        "inner_times": [],
    }
    theta_history = deque(maxlen=5)

    # Create BPTT functions once (JIT cache persists across outer iterations)
    bptt_fns = make_bptt_fns(mjx_model, mj_model, metadata)

    print(f"Configuration:")
    print(f"  Training envs:     {args.num_envs}")
    print(f"  Eval envs:         {args.num_eval_envs}")
    print(f"  Eval horizon:      {args.eval_horizon} steps "
          f"({args.eval_horizon * metadata['timestep']:.1f}s)")
    print(f"  PPO horizon:       {args.horizon} steps")
    print(f"  Design optimizer:  Adam (lr={args.design_lr})")
    print(f"  Design params:     {NUM_DESIGN_PARAMS} (symmetric lower-body)")
    print(f"  Theta bounds:      +/-30 deg (+/-0.5236 rad)")
    print(f"  Max inner samples: {args.max_inner_samples:,}")

    for outer_iter in range(args.outer_iters):
        print(f"\n{'=' * 70}")
        print(f"Outer Iteration {outer_iter + 1}/{args.outer_iters}")
        print(f"{'=' * 70}")

        theta_deg = np.degrees(theta)
        for i, name in enumerate(param_names):
            print(f"  {name}: {theta[i]:+.4f} rad ({theta_deg[i]:+.2f} deg)")

        iter_dir = out_dir / f"outer_{outer_iter:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # ----- Apply theta to model -----
        theta_jax = jnp.array(theta, dtype=jnp.float32)
        mjx_model_modified = apply_theta(
            mjx_model, theta_jax,
            metadata["base_body_quat"],
            metadata["all_param_body_indices"],
            metadata["param_for_body"],
        )

        # ----- Create environment with modified model -----
        env_fns = make_env_fns(mjx_model_modified, metadata)

        # ----- Inner Loop: PPO Training -----
        print(f"\n  [Inner Loop] Training PPO ({args.num_envs} envs)...")
        t0 = time.time()

        rng, rng_inner = jax.random.split(rng)
        (policy_params, value_params, opt_state, obs_rms,
         inner_metrics, converged) = run_inner_loop(
            env_fns=env_fns,
            policy_params=policy_params,
            value_params=value_params,
            opt_state=opt_state,
            ppo_train_fns=ppo_train_fns,
            obs_rms=obs_rms,
            rng=rng_inner,
            ppo_cfg=ppo_cfg,
            gate=convergence_gate,
            max_samples=args.max_inner_samples,
            log_every=args.log_every,
            use_wandb=use_wandb,
            logger=logger,
            mj_model=mj_model,
            video_every=args.video_every,
            env_sharding=env_sharding,
        )

        inner_time = time.time() - t0
        history["inner_times"].append(inner_time)
        print(f"  [Inner Loop] Done in {inner_time / 60:.1f} min "
              f"({inner_metrics['total_samples']:,} samples, "
              f"final_rew={inner_metrics['final_mean_reward']:.4f})")

        # ----- Collect actions for BPTT -----
        print(f"\n  [Actions] Collecting ({args.num_eval_envs} envs x "
              f"{args.eval_horizon} steps)...")

        rng, rng_eval = jax.random.split(rng)

        # Create eval env with same modified model
        eval_env_fns = make_env_fns(mjx_model_modified, metadata)
        actions_batch, init_qpos_batch = collect_eval_actions(
            eval_env_fns, policy_apply, policy_params, obs_rms,
            args.num_eval_envs, args.eval_horizon, rng_eval,
            env_sharding=env_sharding,
        )
        print(f"    Got actions: {actions_batch.shape}")

        # ----- BPTT Gradient -----
        print(f"\n  [BPTT] Computing gradient via jax.grad(mjx.step)...")
        t0_bptt = time.time()

        grad_theta_jax, mean_fwd_dist, mean_cot = compute_bptt_gradient(
            bptt_fns, theta_jax, actions_batch, init_qpos_batch,
        )

        bptt_time = time.time() - t0_bptt
        grad_theta = np.asarray(grad_theta_jax, dtype=np.float64)
        fwd_dist = float(mean_fwd_dist)
        cot = float(mean_cot)

        print(f"  [BPTT] Done in {bptt_time:.1f}s")
        print(f"    BPTT gradients:")
        for i, name in enumerate(param_names):
            print(f"      d_CoT/d_{name} = {grad_theta[i]:+.6f}")
        print(f"    Forward distance = {fwd_dist:.3f} m")
        print(f"    Cost of Transport = {cot:.4f}")

        history["forward_dist"].append(fwd_dist)
        history["cot"].append(cot)
        history["gradients"].append(grad_theta.copy())

        # ----- Design Update -----
        if np.any(np.isnan(grad_theta)) or np.all(grad_theta == 0):
            print(f"\n  [FATAL] Degenerate gradient (NaN or all-zero)")
            break

        # Minimize CoT: reward_grad = -d(CoT)/d(theta)
        reward_grad = -grad_theta
        old_theta = theta.copy()
        theta = design_optimizer.step(theta, reward_grad)
        theta = np.clip(theta, theta_bounds[0], theta_bounds[1])

        print(f"\n  Design update (minimize CoT = ascend -CoT):")
        for i, name in enumerate(param_names):
            delta = theta[i] - old_theta[i]
            print(f"    {name}: {old_theta[i]:+.4f} -> {theta[i]:+.4f} "
                  f"(delta={delta:+.5f}, {np.degrees(delta):+.3f} deg)")

        history["theta"].append(theta.copy())

        # ----- Save checkpoint -----
        np.save(str(out_dir / "theta_latest.npy"), theta)
        np.save(str(iter_dir / "theta.npy"), theta)
        np.save(str(iter_dir / "grad.npy"), grad_theta)

        # Log to wandb
        if use_wandb:
            log_dict = {
                "outer/iteration": outer_iter + 1,
                "outer/eval_forward_distance": fwd_dist,
                "outer/cot": cot,
                "outer/inner_time_min": inner_time / 60.0,
                "outer/bptt_time_s": bptt_time,
                "outer/grad_norm": float(np.linalg.norm(grad_theta)),
                "outer/inner_final_reward": inner_metrics["final_mean_reward"],
            }
            for i, name in enumerate(param_names):
                log_dict[f"outer/{name}_rad"] = theta[i]
                log_dict[f"outer/{name}_deg"] = np.degrees(theta[i])
                log_dict[f"outer/grad_{name}"] = grad_theta[i]
            logger.log(log_dict)

        # ----- Outer convergence check -----
        theta_history.append(theta.copy())
        if len(theta_history) >= 5:
            theta_stack = np.array(list(theta_history))
            ranges = theta_stack.max(axis=0) - theta_stack.min(axis=0)
            max_range = ranges.max()
            if max_range < np.radians(0.5):
                print(f"\n  OUTER CONVERGED: theta stable "
                      f"(max range = {np.degrees(max_range):.3f} deg)")
                break

    # =====================================================================
    # Final results
    # =====================================================================
    print("\n" + "=" * 70)
    print("PGHC Co-Design Complete!")
    print("=" * 70)

    initial = history["theta"][0]
    final = history["theta"][-1]
    for i, name in enumerate(param_names):
        print(f"  {name}: {initial[i]:+.4f} -> {final[i]:+.4f} "
              f"({np.degrees(initial[i]):+.2f} -> "
              f"{np.degrees(final[i]):+.2f} deg)")

    if history["forward_dist"]:
        print(f"\nForward distance: {history['forward_dist'][0]:.3f} -> "
              f"{history['forward_dist'][-1]:.3f} m")
    if history["cot"]:
        print(f"Cost of Transport: {history['cot'][0]:.4f} -> "
              f"{history['cot'][-1]:.4f}")

    total_time = sum(history["inner_times"])
    print(f"Total inner loop time: {total_time / 3600:.1f} hours")

    if use_wandb:
        logger.finish()

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGHC Co-Design for G1 Humanoid (JAX/MJX)"
    )
    # Outer loop
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--design-lr", type=float, default=0.005)
    parser.add_argument("--out-dir", type=str, default="output_g1_jax")

    # Inner loop (PPO)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--ppo-lr", type=float, default=1e-3)
    parser.add_argument("--max-inner-samples", type=int, default=5_000_000_000)
    parser.add_argument("--min-inner-iters", type=int, default=2000)
    parser.add_argument("--stable-iters-required", type=int, default=50)
    parser.add_argument("--plateau-threshold", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--video-every", type=int, default=100)

    # Eval / BPTT
    parser.add_argument("--num-eval-envs", type=int, default=32)
    parser.add_argument("--eval-horizon", type=int, default=50)

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=False)

    args = parser.parse_args()
    main(args)
