#!/usr/bin/env python3
"""
Finite Difference Validation of MJX BPTT Gradients.

Compares jax.grad(CoT) against central finite differences for each
of the 6 morphology parameters (theta). A passing test has FD ratios
close to 1.0, indicating jax.grad through mjx.step is correct.

Tests:
  A: Single rollout, short horizon (5 steps), zero actions
  B: Single rollout, short horizon (5 steps), random actions
  C: Single rollout, medium horizon (20 steps), random actions
  D: Single rollout, medium horizon (20 steps), policy-like actions
  E: Gradient w.r.t. a simple loss (root height after N steps)
  F: Batched gradient (multiple rollouts averaged)
"""

import sys
import os
import time

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from g1_model import load_g1_model, NUM_DESIGN_PARAMS
from g1_morphology import apply_theta
from g1_bptt_jax import make_bptt_fns


def make_simple_loss_fn(mjx_model, mj_model, metadata):
    """Create a simple loss: root height after N steps. Easier to differentiate
    than CoT (no division by distance), good for isolating gradient issues."""

    base_body_quat = metadata["base_body_quat"]
    all_param_body_indices = metadata["all_param_body_indices"]
    param_for_body = metadata["param_for_body"]
    dt = metadata["timestep"]
    default_qpos = metadata["default_qpos"]
    nv = metadata["nv"]
    default_dof_pos_act = default_qpos[7:]
    action_scale = 0.25

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    template_data = mjx.put_data(mj_model, mj_data)

    def height_loss(theta, actions, init_qpos):
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)
        data = template_data.replace(qpos=init_qpos, qvel=jnp.zeros(nv))
        data = mjx.forward(model, data)

        @jax.checkpoint
        def step_fn(data, action):
            target = action * action_scale + default_dof_pos_act
            data = data.replace(ctrl=target)
            data = mjx.step(model, data)
            return data, None

        final_data, _ = jax.lax.scan(step_fn, data, actions)
        return final_data.qpos[2]  # root z height

    return height_loss


def fd_gradient(loss_fn, theta, *args, eps=1e-4):
    """Central finite difference gradient."""
    n = len(theta)
    grad = np.zeros(n)
    for i in range(n):
        theta_plus = theta.at[i].set(theta[i] + eps)
        theta_minus = theta.at[i].set(theta[i] - eps)
        f_plus = float(loss_fn(theta_plus, *args))
        f_minus = float(loss_fn(theta_minus, *args))
        grad[i] = (f_plus - f_minus) / (2 * eps)
        print(f"        FD param {i+1}/{n}: "
              f"f+={f_plus:.8f}, f-={f_minus:.8f}, "
              f"grad={grad[i]:.8f}", flush=True)
    return grad


def run_test(name, loss_fn, theta, actions, init_qpos, eps=1e-4):
    """Run a single gradient validation test."""
    print(f"\n{'='*60}")
    print(f"Test {name}")
    print(f"{'='*60}")

    # Analytical gradient via jax.grad
    print(f"  [1/2] Computing jax.grad (JIT compiling on first call)...",
          flush=True)
    t0 = time.time()
    grad_fn = jax.grad(loss_fn, argnums=0)
    grad_analytical = np.asarray(grad_fn(theta, actions, init_qpos))
    t_analytical = time.time() - t0
    print(f"  [1/2] Done in {t_analytical:.1f}s")

    # Finite difference gradient
    print(f"  [2/2] Computing FD gradient (12 forward passes)...",
          flush=True)
    t0 = time.time()
    grad_fd = fd_gradient(loss_fn, theta, actions, init_qpos, eps=eps)
    t_fd = time.time() - t0
    print(f"  [2/2] Done in {t_fd:.1f}s")

    # Compare
    print(f"\n  {'Param':<8} {'Analytical':>14} {'FD':>14} {'Ratio':>10} {'Status':>8}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10} {'-'*8}")

    all_pass = True
    for i in range(len(theta)):
        a = grad_analytical[i]
        f = grad_fd[i]
        if abs(f) < 1e-10 and abs(a) < 1e-10:
            ratio = 1.0
            status = "SKIP"
        elif abs(f) < 1e-10:
            ratio = float('inf')
            status = "WARN"
            all_pass = False
        else:
            ratio = a / f
            if 0.9 < ratio < 1.1:
                status = "PASS"
            elif 0.5 < ratio < 2.0:
                status = "SOFT"
                all_pass = False
            else:
                status = "FAIL"
                all_pass = False

        print(f"  theta_{i:<3} {a:>14.8f} {f:>14.8f} {ratio:>10.4f} {status:>8}")

    grad_norm_a = np.linalg.norm(grad_analytical)
    grad_norm_f = np.linalg.norm(grad_fd)
    cos_sim = (np.dot(grad_analytical, grad_fd) /
               max(grad_norm_a * grad_norm_f, 1e-12))

    print(f"\n  Gradient norm (analytical): {grad_norm_a:.8f}")
    print(f"  Gradient norm (FD):         {grad_norm_f:.8f}")
    print(f"  Cosine similarity:          {cos_sim:.6f}")
    print(f"  Time (analytical):          {t_analytical:.2f}s")
    print(f"  Time (FD):                  {t_fd:.2f}s")
    print(f"  Overall: {'PASS' if all_pass else 'ISSUES DETECTED'}")

    return all_pass, cos_sim


def main():
    print("=" * 60)
    print("MJX BPTT Gradient Validation (jax.grad vs Finite Differences)")
    print("=" * 60)

    # Suppress XLA warnings
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    # Load model
    print("\nLoading G1 model...")
    mj_model, mjx_model, metadata = load_g1_model()

    nq = metadata["nq"]
    nv = metadata["nv"]
    default_qpos = metadata["default_qpos"]

    # Create BPTT functions (CoT loss)
    bptt_fns = make_bptt_fns(mjx_model, mj_model, metadata)
    cot_loss_fn = bptt_fns["loss_with_info_fn"]

    # Scalar CoT loss (drops info)
    def cot_loss(theta, actions, init_qpos):
        cot, _ = cot_loss_fn(theta, actions, init_qpos)
        return cot

    # Simple height loss
    height_loss_fn = make_simple_loss_fn(mjx_model, mj_model, metadata)

    # Base theta (small nonzero to avoid symmetric point)
    rng = jax.random.PRNGKey(123)
    theta = jnp.array([0.05, -0.03, 0.02, -0.04, 0.01, -0.02])
    init_qpos = default_qpos

    # Warmup: JIT compile the forward pass with a tiny rollout
    print("\nWarmup: JIT compiling forward pass...", flush=True)
    t0 = time.time()
    warmup_actions = jnp.zeros((3, metadata["num_actuators"]))
    _ = height_loss_fn(theta, warmup_actions, init_qpos)
    print(f"Warmup forward done in {time.time() - t0:.1f}s", flush=True)

    print("Warmup: JIT compiling jax.grad...", flush=True)
    t0 = time.time()
    _ = jax.grad(height_loss_fn)(theta, warmup_actions, init_qpos)
    print(f"Warmup grad done in {time.time() - t0:.1f}s", flush=True)

    results = []

    # ----- Test A: Height loss, 5 steps, zero actions -----
    actions_zero_5 = jnp.zeros((5, metadata["num_actuators"]))
    ok, cos = run_test("A: Height loss, 5 steps, zero actions",
                       height_loss_fn, theta, actions_zero_5, init_qpos)
    results.append(("A", ok, cos))

    # ----- Test B: Height loss, 5 steps, random actions -----
    rng, rng_act = jax.random.split(rng)
    actions_rand_5 = jax.random.uniform(rng_act,
                                         (5, metadata["num_actuators"]),
                                         minval=-0.5, maxval=0.5)
    ok, cos = run_test("B: Height loss, 5 steps, random actions",
                       height_loss_fn, theta, actions_rand_5, init_qpos)
    results.append(("B", ok, cos))

    # ----- Test C: Height loss, 20 steps, random actions -----
    rng, rng_act = jax.random.split(rng)
    actions_rand_20 = jax.random.uniform(rng_act,
                                          (20, metadata["num_actuators"]),
                                          minval=-0.3, maxval=0.3)
    ok, cos = run_test("C: Height loss, 20 steps, random actions",
                       height_loss_fn, theta, actions_rand_20, init_qpos)
    results.append(("C", ok, cos))

    # ----- Test D: CoT loss, 5 steps, zero actions -----
    ok, cos = run_test("D: CoT loss, 5 steps, zero actions",
                       cot_loss, theta, actions_zero_5, init_qpos)
    results.append(("D", ok, cos))

    # ----- Test E: CoT loss, 5 steps, random actions -----
    ok, cos = run_test("E: CoT loss, 5 steps, random actions",
                       cot_loss, theta, actions_rand_5, init_qpos)
    results.append(("E", ok, cos))

    # ----- Test F: CoT loss, 20 steps, random actions -----
    ok, cos = run_test("F: CoT loss, 20 steps, random actions",
                       cot_loss, theta, actions_rand_20, init_qpos,
                       eps=5e-5)
    results.append(("F", ok, cos))

    # ----- Summary -----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok, cos in results:
        status = "PASS" if ok else "ISSUES"
        print(f"  Test {name}: {status:>8}  (cos_sim={cos:.6f})")

    num_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {num_pass}/{len(results)} tests passed")

    if num_pass == len(results):
        print("\n  All gradients validated! MJX BPTT is trustworthy.")
    else:
        print("\n  Some gradients have issues. Investigate before trusting BPTT.")

    return num_pass == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
