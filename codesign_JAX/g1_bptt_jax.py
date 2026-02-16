"""
BPTT Gradient Computation via Finite Differences through mjx.step.

Core differentiable physics evaluation for PGHC outer loop.
Computes Cost of Transport (CoT) gradients w.r.t. morphology parameters theta.

Uses central finite differences instead of jax.grad because MJX's constraint
solver does not correctly propagate gradients through model parameters
(body_quat). See test_gradient_diagnostic.py for validation. With only 6
design parameters, FD requires just 12 forward passes (~1s total after JIT).
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from g1_morphology import apply_theta


def make_bptt_fns(mjx_model, mj_model, metadata):
    """Create BPTT loss and gradient functions.

    Args:
        mjx_model: mjx.Model (base, before theta applied)
        mj_model: mujoco.MjModel (for creating initial data)
        metadata: dict from g1_model.load_g1_model()

    Returns:
        dict with 'batched_loss_fn', 'loss_with_info_fn'
    """
    base_body_quat = metadata["base_body_quat"]
    all_param_body_indices = metadata["all_param_body_indices"]
    param_for_body = metadata["param_for_body"]
    total_mass = metadata["total_mass"]
    dt = metadata["timestep"]
    default_qpos = metadata["default_qpos"]
    nv = metadata["nv"]

    # Default actuated DOF positions (for action scaling)
    default_dof_pos_act = default_qpos[7:]
    action_scale = 0.25  # matches env config

    # Create template data on CPU, then put to device once
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    template_data = mjx.put_data(mj_model, mj_data)

    def bptt_loss_with_info(theta, actions, init_qpos):
        """Compute CoT through physics simulation, with auxiliary info.

        Args:
            theta: (6,) design parameters
            actions: (H, num_act) action sequence
            init_qpos: (nq,) initial joint positions

        Returns:
            cot: scalar Cost of Transport
            info: dict with forward_dist, energy, cot, final_root_z
        """
        # 1. Apply theta to model
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)

        # 2. Initialize state
        data = template_data.replace(qpos=init_qpos, qvel=jnp.zeros(nv))
        data = mjx.forward(model, data)

        initial_x = data.qpos[0]

        # 3. Forward pass through physics
        def step_fn(data, action):
            target = action * action_scale + default_dof_pos_act
            data = data.replace(ctrl=target)
            data = mjx.step(model, data)

            # Accumulate actuated DOF velocity energy
            act_vel = data.qvel[6:]
            energy = jnp.sum(act_vel**2) * dt
            return data, energy

        final_data, energies = jax.lax.scan(step_fn, data, actions)

        # 4. Compute CoT = Energy / (m * g * d)
        forward_dist = final_data.qpos[0] - initial_x
        total_energy = jnp.sum(energies)

        # Clamp forward distance to avoid division by zero
        safe_dist = jnp.maximum(forward_dist, 0.01)
        cot = total_energy / (total_mass * 9.81 * safe_dist)

        info = {
            "forward_dist": forward_dist,
            "total_energy": total_energy,
            "cot": cot,
            "final_root_z": final_data.qpos[2],
        }
        return cot, info

    # Batched: vmap over (actions, init_qpos), shared theta
    batched_loss_fn = jax.jit(jax.vmap(bptt_loss_with_info, in_axes=(None, 0, 0)))

    return {
        "loss_with_info_fn": bptt_loss_with_info,
        "batched_loss_fn": batched_loss_fn,
    }


def compute_bptt_gradient(bptt_fns, theta, actions_batch, init_qpos_batch,
                           eps=1e-4):
    """Compute mean CoT gradient via central finite differences.

    With 6 design parameters, this requires 12 forward passes (plus 1
    baseline). Each forward pass is JIT-compiled and vmapped over eval
    envs, so the total time is ~1s after warmup.

    Args:
        bptt_fns: dict from make_bptt_fns()
        theta: (6,) design parameters
        actions_batch: (B, H, num_act) action sequences
        init_qpos_batch: (B, nq) initial configurations
        eps: finite difference step size

    Returns:
        mean_grad: (6,) mean gradient
        mean_fwd_dist: scalar mean forward distance
        mean_cot: scalar mean Cost of Transport
    """
    batched_loss_fn = bptt_fns["batched_loss_fn"]
    n_params = len(theta)

    # Baseline evaluation (for info only — not needed for central FD)
    cot_values, infos = batched_loss_fn(theta, actions_batch, init_qpos_batch)
    mean_cot = float(jnp.mean(cot_values))
    mean_fwd_dist = float(jnp.mean(infos["forward_dist"]))

    # Central finite differences
    grad = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        theta_plus = theta.at[i].set(theta[i] + eps)
        theta_minus = theta.at[i].set(theta[i] - eps)

        cot_plus, _ = batched_loss_fn(theta_plus, actions_batch, init_qpos_batch)
        cot_minus, _ = batched_loss_fn(theta_minus, actions_batch, init_qpos_batch)

        grad[i] = (float(jnp.mean(cot_plus)) - float(jnp.mean(cot_minus))) / (2 * eps)

    # Guard against NaN/Inf
    if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
        grad = np.zeros_like(grad)

    # Clip gradients
    grad = np.clip(grad, -10.0, 10.0)

    return jnp.array(grad, dtype=jnp.float32), mean_fwd_dist, mean_cot
