"""
BPTT Gradient Computation via jax.grad through mjx.step.

Core differentiable physics evaluation for PGHC outer loop.
Computes Cost of Transport (CoT) gradients w.r.t. morphology parameters theta.
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from g1_morphology import apply_theta


def make_bptt_fns(mjx_model, mj_model, metadata):
    """Create BPTT loss and gradient functions.

    Args:
        mjx_model: mjx.Model (base, before theta applied)
        mj_model: mujoco.MjModel (for creating initial data)
        metadata: dict from g1_model.load_g1_model()

    Returns:
        dict with 'loss_fn', 'grad_fn', 'compute_gradient'
    """
    base_body_quat = metadata["base_body_quat"]
    all_param_body_indices = metadata["all_param_body_indices"]
    param_for_body = metadata["param_for_body"]
    total_mass = metadata["total_mass"]
    dt = metadata["timestep"]
    default_qpos = metadata["default_qpos"]
    nv = metadata["nv"]
    nq = metadata["nq"]

    # Default actuated DOF positions (for action scaling)
    default_dof_pos_act = default_qpos[7:]
    action_scale = 0.25  # matches env config

    # Create template data on CPU, then put to device once
    # (avoids calling mjx.make_data inside traced code)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    template_data = mjx.put_data(mj_model, mj_data)

    def bptt_loss_with_info(theta, actions, init_qpos):
        """Compute CoT through differentiable physics, with auxiliary info.

        Args:
            theta: (6,) design parameters
            actions: (H, num_act) action sequence
            init_qpos: (nq,) initial joint positions

        Returns:
            cot: scalar Cost of Transport
            info: dict with forward_dist, energy, cot, final_root_z
        """
        # 1. Apply theta to model (differentiable)
        model = apply_theta(mjx_model, theta, base_body_quat,
                           all_param_body_indices, param_for_body)

        # 2. Initialize state from template (avoid mjx.make_data in traced code)
        data = template_data.replace(qpos=init_qpos, qvel=jnp.zeros(nv))
        data = mjx.forward(model, data)

        initial_x = data.qpos[0]

        # 3. Forward pass through physics with gradient checkpointing
        @jax.checkpoint
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

    def bptt_loss(theta, actions, init_qpos):
        """Scalar loss for jax.grad (drops info)."""
        cot, _ = bptt_loss_with_info(theta, actions, init_qpos)
        return cot

    # Single-rollout gradient + value (one forward + one backward pass)
    grad_and_value_fn = jax.value_and_grad(bptt_loss, argnums=0)

    # Batched: vmap over (actions, init_qpos), shared theta
    batched_grad_and_value_fn = jax.vmap(
        grad_and_value_fn, in_axes=(None, 0, 0)
    )
    batched_loss_with_info = jax.vmap(
        bptt_loss_with_info, in_axes=(None, 0, 0)
    )

    return {
        "loss_fn": bptt_loss,
        "loss_with_info_fn": bptt_loss_with_info,
        "grad_and_value_fn": grad_and_value_fn,
        "batched_grad_and_value_fn": batched_grad_and_value_fn,
        "batched_loss_with_info_fn": batched_loss_with_info,
    }


def compute_bptt_gradient(bptt_fns, theta, actions_batch, init_qpos_batch):
    """Compute mean BPTT gradient over a batch of rollouts.

    Uses value_and_grad to avoid double forward pass.

    Args:
        bptt_fns: dict from make_bptt_fns()
        theta: (6,) design parameters
        actions_batch: (B, H, num_act) action sequences
        init_qpos_batch: (B, nq) initial configurations

    Returns:
        mean_grad: (6,) mean gradient (numpy)
        mean_fwd_dist: scalar mean forward distance
        mean_cot: scalar mean Cost of Transport
    """
    # Compute per-rollout CoT values and gradients in one pass
    cot_values, grads = bptt_fns["batched_grad_and_value_fn"](
        theta, actions_batch, init_qpos_batch
    )

    # Also get info (forward distances etc.) - this is a second forward pass
    # but it's cheap relative to the backward pass
    _, infos = bptt_fns["batched_loss_with_info_fn"](
        theta, actions_batch, init_qpos_batch
    )

    # Average gradients across batch
    mean_grad = jnp.mean(grads, axis=0)
    mean_fwd_dist = jnp.mean(infos["forward_dist"])
    mean_cot = jnp.mean(cot_values)

    # Guard against NaN/Inf
    has_bad = jnp.any(jnp.isnan(mean_grad)) | jnp.any(jnp.isinf(mean_grad))
    mean_grad = jnp.where(has_bad, jnp.zeros_like(mean_grad), mean_grad)

    # Clip gradients
    mean_grad = jnp.clip(mean_grad, -10.0, 10.0)

    return mean_grad, mean_fwd_dist, mean_cot
