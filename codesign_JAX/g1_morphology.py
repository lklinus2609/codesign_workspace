"""
Differentiable Theta -> Body Quaternion Mapping for G1 Morphology.

Ports g1_mjcf_modifier.py from XML-based (non-differentiable) to pure JAX
operations that are compatible with jax.grad.

Each theta[i] adds an X-axis rotation to the body frame quaternion of one
symmetric pair (left + right). The mapping is:
    new_quat = quat_from_x_rotation(theta[i]) * base_quat

MuJoCo/MJX quaternion convention: (w, x, y, z)
"""

import jax
import jax.numpy as jnp


def quat_from_x_rotation(angle):
    """Quaternion for rotation around X axis. Returns (w, x, y, z).

    Args:
        angle: scalar rotation angle in radians

    Returns:
        (4,) quaternion array in wxyz format
    """
    half = angle * 0.5
    return jnp.array([jnp.cos(half), jnp.sin(half), 0.0, 0.0])


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2. Both in (w, x, y, z) format.

    Args:
        q1, q2: (4,) quaternion arrays

    Returns:
        (4,) quaternion array
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_normalize(q):
    """Normalize quaternion (w, x, y, z).

    Args:
        q: (4,) quaternion array

    Returns:
        (4,) normalized quaternion
    """
    norm = jnp.sqrt(jnp.sum(q * q))
    return q / jnp.maximum(norm, 1e-12)


def apply_theta(mjx_model, theta, base_body_quat, all_param_body_indices,
                param_for_body):
    """Apply design parameters theta to MJX model body quaternions.

    Pure JAX function, compatible with jax.grad.

    Args:
        mjx_model: mjx.Model (JAX PyTree)
        theta: (6,) array of design angles in radians
        base_body_quat: (nbody, 4) original body quaternions (wxyz)
        all_param_body_indices: (12,) int array of body indices to modify
        param_for_body: (12,) int array mapping each body to its theta index

    Returns:
        Modified mjx_model with updated body_quat
    """
    body_quat = base_body_quat  # (nbody, 4) - start from base

    def update_one_body(body_quat, i):
        """Update one parameterized body's quaternion."""
        body_idx = all_param_body_indices[i]
        theta_idx = param_for_body[i]
        angle = theta[theta_idx]

        # Compute delta quaternion from X-rotation
        delta_q = quat_from_x_rotation(angle)

        # Hamilton product: delta * base
        base_q = base_body_quat[body_idx]
        new_q = quat_normalize(quat_multiply(delta_q, base_q))

        # Update the body_quat array
        body_quat = body_quat.at[body_idx].set(new_q)
        return body_quat, None

    # Apply to all 12 parameterized bodies
    n_param_bodies = all_param_body_indices.shape[0]
    body_quat, _ = jax.lax.scan(update_one_body, body_quat,
                                 jnp.arange(n_param_bodies))

    return mjx_model.replace(body_quat=body_quat)
