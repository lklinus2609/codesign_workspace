#!/usr/bin/env python3
"""
Finite Difference Validation of MJX BPTT Gradients.

Compares jax.grad(loss) against central finite differences for each
of the 6 morphology parameters (theta). A passing test has FD ratios
close to 1.0, indicating jax.grad through mjx.step is correct.

All tests use a FIXED horizon (20 steps) to avoid repeated JIT
compilations — each unique jax.lax.scan length triggers a full XLA
recompilation (~2-3 min). By keeping the horizon constant, we compile
jax.grad exactly once per loss function.

Tests:
  A: Height loss, zero actions
  B: Height loss, small random actions
  C: Height loss, larger random actions
  D: CoT loss, zero actions
  E: CoT loss, small random actions
  F: CoT loss, larger random actions
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

# Fixed horizon for ALL tests (avoids recompilation)
HORIZON = 20


def make_simple_loss_fn(mjx_model, mj_model, metadata):
    """Create a simple loss: root height after N steps. Easier to differentiate
    than CoT (no division by distance), good for isolating gradient issues."""

    base_body_quat = metadata["base_body_quat"]
    all_param_body_indices = metadata["all_param_body_indices"]
    param_for_body = metadata["param_for_body"]
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


def run_test(name, grad_fn_jit, loss_fn_jit, theta, actions, init_qpos,
             eps=1e-4):
    """Run a single gradient validation test using pre-compiled functions."""
    print(f"\n{'='*60}")
    print(f"Test {name}")
    print(f"{'='*60}")

    # Analytical gradient via pre-compiled jax.grad
    print(f"  [1/2] Computing jax.grad (already JIT compiled)...", flush=True)
    t0 = time.time()
    grad_analytical = np.asarray(grad_fn_jit(theta, actions, init_qpos))
    t_analytical = time.time() - t0
    print(f"  [1/2] Done in {t_analytical:.1f}s")

    # Finite difference gradient using pre-compiled forward fn
    print(f"  [2/2] Computing FD gradient (12 forward passes)...", flush=True)
    t0 = time.time()
    grad_fd = fd_gradient(loss_fn_jit, theta, actions, init_qpos, eps=eps)
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
    print(f"Fixed horizon: {HORIZON} steps (all tests use same shape "
          f"to avoid recompilation)")

    # Load model
    print("\nLoading G1 model...")
    mj_model, mjx_model, metadata = load_g1_model()

    default_qpos = metadata["default_qpos"]
    num_act = metadata["num_actuators"]

    # Create BPTT functions (CoT loss)
    bptt_fns = make_bptt_fns(mjx_model, mj_model, metadata)
    cot_loss_info_fn = bptt_fns["loss_with_info_fn"]

    # Scalar CoT loss (drops info)
    def cot_loss(theta, actions, init_qpos):
        cot, _ = cot_loss_info_fn(theta, actions, init_qpos)
        return cot

    # Simple height loss
    height_loss_fn = make_simple_loss_fn(mjx_model, mj_model, metadata)

    # Base theta (small nonzero to avoid symmetric point)
    rng = jax.random.PRNGKey(123)
    theta = jnp.array([0.05, -0.03, 0.02, -0.04, 0.01, -0.02])
    init_qpos = default_qpos

    # ---------------------------------------------------------------
    # One-time JIT compilation for BOTH loss functions at fixed horizon
    # This is the slow part (~2-3 min each), but only happens ONCE.
    # ---------------------------------------------------------------
    warmup_actions = jnp.zeros((HORIZON, num_act))

    print(f"\n[Compile 1/4] JIT compiling height_loss forward "
          f"({HORIZON} steps)...", flush=True)
    t0 = time.time()
    height_loss_jit = jax.jit(height_loss_fn)
    _ = height_loss_jit(theta, warmup_actions, init_qpos).block_until_ready()
    print(f"[Compile 1/4] Done in {time.time() - t0:.1f}s", flush=True)

    print(f"[Compile 2/4] JIT compiling jax.grad(height_loss) "
          f"({HORIZON} steps)...", flush=True)
    t0 = time.time()
    height_grad_jit = jax.jit(jax.grad(height_loss_fn, argnums=0))
    _ = height_grad_jit(theta, warmup_actions, init_qpos).block_until_ready()
    print(f"[Compile 2/4] Done in {time.time() - t0:.1f}s", flush=True)

    print(f"[Compile 3/4] JIT compiling cot_loss forward "
          f"({HORIZON} steps)...", flush=True)
    t0 = time.time()
    cot_loss_jit = jax.jit(cot_loss)
    _ = cot_loss_jit(theta, warmup_actions, init_qpos).block_until_ready()
    print(f"[Compile 3/4] Done in {time.time() - t0:.1f}s", flush=True)

    print(f"[Compile 4/4] JIT compiling jax.grad(cot_loss) "
          f"({HORIZON} steps)...", flush=True)
    t0 = time.time()
    cot_grad_jit = jax.jit(jax.grad(cot_loss, argnums=0))
    _ = cot_grad_jit(theta, warmup_actions, init_qpos).block_until_ready()
    print(f"[Compile 4/4] Done in {time.time() - t0:.1f}s", flush=True)

    print("\nAll JIT compilations done. Running tests (no more recompilation).")

    results = []

    # Generate action arrays (all same shape = HORIZON x num_act)
    actions_zero = jnp.zeros((HORIZON, num_act))

    rng, rng_a, rng_b = jax.random.split(rng, 3)
    actions_small = jax.random.uniform(rng_a, (HORIZON, num_act),
                                        minval=-0.3, maxval=0.3)
    actions_large = jax.random.uniform(rng_b, (HORIZON, num_act),
                                        minval=-0.8, maxval=0.8)

    # ----- Test A: Height loss, zero actions -----
    ok, cos = run_test("A: Height loss, zero actions",
                       height_grad_jit, height_loss_jit,
                       theta, actions_zero, init_qpos)
    results.append(("A", ok, cos))

    # ----- Test B: Height loss, small random actions -----
    ok, cos = run_test("B: Height loss, small random actions",
                       height_grad_jit, height_loss_jit,
                       theta, actions_small, init_qpos)
    results.append(("B", ok, cos))

    # ----- Test C: Height loss, large random actions -----
    ok, cos = run_test("C: Height loss, large random actions",
                       height_grad_jit, height_loss_jit,
                       theta, actions_large, init_qpos)
    results.append(("C", ok, cos))

    # ----- Test D: CoT loss, zero actions -----
    ok, cos = run_test("D: CoT loss, zero actions",
                       cot_grad_jit, cot_loss_jit,
                       theta, actions_zero, init_qpos)
    results.append(("D", ok, cos))

    # ----- Test E: CoT loss, small random actions -----
    ok, cos = run_test("E: CoT loss, small random actions",
                       cot_grad_jit, cot_loss_jit,
                       theta, actions_small, init_qpos)
    results.append(("E", ok, cos))

    # ----- Test F: CoT loss, large random actions -----
    ok, cos = run_test("F: CoT loss, large random actions",
                       cot_grad_jit, cot_loss_jit,
                       theta, actions_large, init_qpos,
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
