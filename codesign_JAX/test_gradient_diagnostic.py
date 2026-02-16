#!/usr/bin/env python3
"""
Gradient Diagnostic: isolates WHERE jax.grad through MJX breaks down.

Runs a series of increasingly complex tests to pinpoint the gradient issue:
  1. Pure quaternion math (should be perfect)
  2. apply_theta -> model.body_quat (tests PyTree replace)
  3. mjx.forward only (tests FK gradient w.r.t. model params)
  4. 1 mjx.step (tests single step gradient)
  5. 20 mjx.steps WITHOUT jax.checkpoint
  6. 20 mjx.steps WITH jax.checkpoint
  7. 20 mjx.steps WITH checkpoint, float64

Each test compares jax.grad vs central FD. The first test that shows
a magnitude mismatch is where the problem lies.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must set before importing JAX
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from g1_model import load_g1_model, NUM_DESIGN_PARAMS
from g1_morphology import apply_theta, quat_from_x_rotation, quat_multiply


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
    return grad


def compare(name, analytical, fd, compile_time=None):
    """Print comparison table."""
    print(f"\n  {'Param':<8} {'Analytical':>14} {'FD':>14} {'Ratio':>10} {'Status':>8}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10} {'-'*8}")

    all_pass = True
    for i in range(len(analytical)):
        a, f = analytical[i], fd[i]
        if abs(f) < 1e-12 and abs(a) < 1e-12:
            ratio, status = 1.0, "SKIP"
        elif abs(f) < 1e-12:
            ratio, status = float('inf'), "WARN"
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

    norm_a = np.linalg.norm(analytical)
    norm_f = np.linalg.norm(fd)
    cos_sim = np.dot(analytical, fd) / max(norm_a * norm_f, 1e-12)
    mean_ratio = np.mean([a/f for a, f in zip(analytical, fd) if abs(f) > 1e-12]
                         ) if any(abs(f) > 1e-12 for f in fd) else float('nan')

    print(f"\n  ||grad_analytical||: {norm_a:.10f}")
    print(f"  ||grad_fd||:         {norm_f:.10f}")
    print(f"  Cosine similarity:   {cos_sim:.6f}")
    print(f"  Mean ratio (a/fd):   {mean_ratio:.6f}")
    if compile_time:
        print(f"  Compile time:        {compile_time:.1f}s")
    print(f"  Result: {'PASS' if all_pass else 'ISSUES'}")
    return all_pass


def run_diagnostic(name, loss_fn, theta, *extra_args, eps=1e-4):
    """Compile, run grad, run FD, compare."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # JIT compile grad
    print(f"  Compiling jax.grad...", flush=True)
    t0 = time.time()
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
    analytical = np.asarray(grad_fn(theta, *extra_args).block_until_ready())
    compile_time = time.time() - t0
    print(f"  Compiled + ran in {compile_time:.1f}s", flush=True)

    # FD
    print(f"  Computing FD...", flush=True)
    loss_fn_jit = jax.jit(loss_fn)
    _ = loss_fn_jit(theta, *extra_args).block_until_ready()  # warmup
    fd = fd_gradient(loss_fn_jit, theta, *extra_args, eps=eps)

    return compare(name, analytical, fd, compile_time)


def main():
    print("=" * 60)
    print("MJX Gradient Diagnostic")
    print("=" * 60)

    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Float precision: {jnp.zeros(1).dtype}")

    # Load model
    print("\nLoading G1 model...")
    mj_model, mjx_model, metadata = load_g1_model()

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

    theta = jnp.array([0.05, -0.03, 0.02, -0.04, 0.01, -0.02])
    init_qpos = default_qpos

    rng = jax.random.PRNGKey(42)
    actions_20 = jax.random.uniform(rng, (20, metadata["num_actuators"]),
                                     minval=-0.3, maxval=0.3)

    results = []

    # ---- Test 1: Pure quaternion math ----
    first_body_idx = int(all_param_body_indices[0])

    def quat_loss(theta):
        delta_q = quat_from_x_rotation(theta[0])
        base_q = base_body_quat[first_body_idx]
        new_q = quat_multiply(delta_q, base_q)
        return new_q[1]  # x component

    results.append(("1: Quat math", run_diagnostic(
        "Test 1: Pure quaternion math", quat_loss, theta)))

    # ---- Test 2: apply_theta -> body_quat ----
    def body_quat_loss(theta):
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)
        return model.body_quat[first_body_idx, 1]  # x component

    results.append(("2: apply_theta", run_diagnostic(
        "Test 2: apply_theta -> body_quat[idx, 1]", body_quat_loss, theta)))

    # ---- Test 3: mjx.forward -> xpos ----
    # Root body is typically index 1 (index 0 is world)
    root_body_idx = 1

    def forward_xpos_loss(theta, init_qpos):
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)
        data = template_data.replace(qpos=init_qpos, qvel=jnp.zeros(nv))
        data = mjx.forward(model, data)
        return data.xpos[root_body_idx, 2]  # root Z in Cartesian

    results.append(("3: mjx.forward->xpos", run_diagnostic(
        "Test 3: mjx.forward -> xpos[root, z]",
        forward_xpos_loss, theta, init_qpos)))

    # ---- Test 3b: mjx.forward -> subtree_com ----
    def forward_com_loss(theta, init_qpos):
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)
        data = template_data.replace(qpos=init_qpos, qvel=jnp.zeros(nv))
        data = mjx.forward(model, data)
        return data.subtree_com[0, 2]  # world subtree COM Z

    results.append(("3b: mjx.forward->com", run_diagnostic(
        "Test 3b: mjx.forward -> subtree_com[world, z]",
        forward_com_loss, theta, init_qpos)))

    # ---- Test 4: 1 mjx.step -> qpos[2] ----
    def one_step_loss(theta, actions, init_qpos):
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)
        data = template_data.replace(qpos=init_qpos, qvel=jnp.zeros(nv))
        data = mjx.forward(model, data)
        target = actions[0] * action_scale + default_dof_pos_act
        data = data.replace(ctrl=target)
        data = mjx.step(model, data)
        return data.qpos[2]

    results.append(("4: 1 step", run_diagnostic(
        "Test 4: 1 mjx.step -> qpos[2]",
        one_step_loss, theta, actions_20, init_qpos)))

    # ---- Test 5: 20 steps WITHOUT checkpoint ----
    def scan_no_checkpoint_loss(theta, actions, init_qpos):
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)
        data = template_data.replace(qpos=init_qpos, qvel=jnp.zeros(nv))
        data = mjx.forward(model, data)

        def step_fn(data, action):
            target = action * action_scale + default_dof_pos_act
            data = data.replace(ctrl=target)
            data = mjx.step(model, data)
            return data, None

        final_data, _ = jax.lax.scan(step_fn, data, actions)
        return final_data.qpos[2]

    results.append(("5: 20 steps (no ckpt)", run_diagnostic(
        "Test 5: 20 steps WITHOUT jax.checkpoint -> qpos[2]",
        scan_no_checkpoint_loss, theta, actions_20, init_qpos)))

    # ---- Test 6: 20 steps WITH checkpoint ----
    def scan_checkpoint_loss(theta, actions, init_qpos):
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
        return final_data.qpos[2]

    results.append(("6: 20 steps (ckpt)", run_diagnostic(
        "Test 6: 20 steps WITH jax.checkpoint -> qpos[2]",
        scan_checkpoint_loss, theta, actions_20, init_qpos)))

    # ---- Test 7: Float64 (if available) ----
    try:
        jax.config.update("jax_enable_x64", True)
        theta64 = theta.astype(jnp.float64)
        init_qpos64 = init_qpos.astype(jnp.float64)
        actions64 = actions_20.astype(jnp.float64)

        # Need to reload model in float64
        print("\n  Reloading model for float64...", flush=True)
        base_body_quat64 = base_body_quat.astype(jnp.float64)

        def scan_checkpoint_loss_f64(theta, actions, init_qpos):
            model = apply_theta(mjx_model, theta, base_body_quat64,
                               all_param_body_indices, param_for_body)
            data = template_data.replace(
                qpos=init_qpos, qvel=jnp.zeros(nv, dtype=jnp.float64))
            data = mjx.forward(model, data)

            @jax.checkpoint
            def step_fn(data, action):
                target = action * action_scale + default_dof_pos_act.astype(
                    jnp.float64)
                data = data.replace(ctrl=target)
                data = mjx.step(model, data)
                return data, None

            final_data, _ = jax.lax.scan(step_fn, data, actions)
            return final_data.qpos[2]

        results.append(("7: 20 steps f64", run_diagnostic(
            "Test 7: 20 steps WITH checkpoint, float64 -> qpos[2]",
            scan_checkpoint_loss_f64, theta64, actions64, init_qpos64,
            eps=1e-6)))
    except Exception as e:
        print(f"\n  Test 7 (float64) skipped: {e}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    for name, passed in results:
        print(f"  {name:<30} {'PASS' if passed else 'FAIL'}")

    print("\nInterpretation:")
    print("  - If Test 1-2 pass but 3 fails: mjx.forward doesn't propagate")
    print("    model param gradients through FK")
    print("  - If Test 3 passes but 4 fails: mjx.step doesn't propagate")
    print("    model param gradients")
    print("  - If Test 4 passes but 5 fails: gradient degrades over")
    print("    multiple steps (chaos/precision)")
    print("  - If Test 5 passes but 6 fails: jax.checkpoint is the issue")
    print("  - If Test 6 fails but 7 passes: float32 precision issue")


if __name__ == "__main__":
    main()
