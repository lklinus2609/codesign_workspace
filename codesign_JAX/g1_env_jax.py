"""
G1 Walking Environment in Pure JAX.

Vectorized via jax.vmap. All reward computation is pure JAX functions
(no Python control flow on data) for jax.jit + jax.vmap compatibility.

Observation (11 + 3*num_act_dofs):
    [ang_vel(3), projected_gravity(3), commands(3),
     dof_pos_relative(N), dof_vel(N), prev_actions(N),
     sin_phase(1), cos_phase(1)]

Rewards (15 terms from G1RoughCfg):
    tracking_lin_vel, tracking_ang_vel, lin_vel_z, ang_vel_xy,
    orientation, base_height, dof_acc, dof_vel, action_rate,
    dof_pos_limits, alive, hip_pos, contact_no_vel,
    feet_swing_height, contact (phase-aware)

Termination: pelvis contact with ground OR episode timeout (20s)
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Per-environment state carried between steps."""
    mjx_data: mjx.Data          # MJX physics state
    episode_step: jnp.ndarray   # (num_envs,) int32
    commands: jnp.ndarray       # (num_envs, 3)
    prev_actions: jnp.ndarray   # (num_envs, num_act)
    last_actions: jnp.ndarray   # (num_envs, num_act)
    last_dof_vel: jnp.ndarray   # (num_envs, nv-6)
    rng: jnp.ndarray            # (num_envs, 2) PRNGKeys


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class EnvConfig(NamedTuple):
    """Static environment configuration."""
    # Observation scales
    obs_scale_ang_vel: float = 0.25
    obs_scale_dof_pos: float = 1.0
    obs_scale_dof_vel: float = 0.05
    obs_scale_lin_vel: float = 2.0

    # Commands
    cmd_x_vel_min: float = 0.3
    cmd_x_vel_max: float = 1.0
    cmd_y_vel_min: float = -0.3
    cmd_y_vel_max: float = 0.3
    cmd_yaw_vel_min: float = -0.5
    cmd_yaw_vel_max: float = 0.5
    cmd_resample_time: float = 10.0  # seconds

    # Control
    action_scale: float = 0.25

    # Reward scales
    reward_tracking_lin_vel: float = 1.0
    reward_tracking_ang_vel: float = 0.5
    reward_lin_vel_z: float = -2.0
    reward_ang_vel_xy: float = -0.05
    reward_orientation: float = -1.0
    reward_base_height: float = -10.0
    reward_dof_acc: float = -2.5e-7
    reward_dof_vel: float = -1e-3
    reward_action_rate: float = -0.01
    reward_dof_pos_limits: float = -5.0
    reward_alive: float = 0.15
    reward_hip_pos: float = -1.0
    reward_contact_no_vel: float = -0.2
    reward_feet_swing_height: float = -20.0
    reward_contact: float = 0.18

    # Reward parameters
    tracking_sigma: float = 0.25
    base_height_target: float = 0.78
    soft_dof_pos_limit: float = 0.9
    feet_swing_height_target: float = 0.08

    # Termination
    max_episode_length_s: float = 20.0

    # Phase (gait timing)
    phase_period: float = 0.8
    phase_offset: float = 0.5


# ---------------------------------------------------------------------------
# Quaternion utilities (MuJoCo wxyz convention)
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q.

    Args:
        q: (..., 4) quaternion in (w, x, y, z) format
        v: (..., 3) vector in world frame

    Returns:
        (..., 3) vector in local frame
    """
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = jnp.cross(q_vec, v) * q_w * 2.0
    c = q_vec * jnp.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


# ---------------------------------------------------------------------------
# Environment functions
# ---------------------------------------------------------------------------

def make_env_fns(mjx_model, metadata, cfg=None):
    """Create vectorized environment step/reset functions.

    Args:
        mjx_model: mjx.Model
        metadata: dict from g1_model.load_g1_model()
        cfg: EnvConfig (optional)

    Returns:
        dict with 'reset', 'step' functions and env info
    """
    if cfg is None:
        cfg = EnvConfig()

    dt = metadata["timestep"]
    nv = metadata["nv"]
    nq = metadata["nq"]
    nu = metadata["num_actuators"]
    default_qpos = metadata["default_qpos"]
    dof_pos_lower = metadata["dof_pos_lower"]
    dof_pos_upper = metadata["dof_pos_upper"]
    feet_body_indices = metadata["feet_body_indices"]
    pelvis_body_index = metadata["pelvis_body_index"]
    hip_dof_indices = metadata["hip_dof_indices"]
    total_mass = metadata["total_mass"]

    # Actuated DOFs start after the 6 root DOFs (freejoint)
    num_act_dofs = nv - 6
    obs_dim = 3 + 3 + 3 + 3 * num_act_dofs + 2

    max_episode_steps = int(cfg.max_episode_length_s / dt)
    cmd_resample_steps = max(1, int(cfg.cmd_resample_time / dt))

    # Soft DOF limits
    mid = (dof_pos_upper + dof_pos_lower) / 2
    half_range = (dof_pos_upper - dof_pos_lower) / 2
    soft_lower = mid - cfg.soft_dof_pos_limit * half_range
    soft_upper = mid + cfg.soft_dof_pos_limit * half_range
    # Only care about actuated DOFs (skip 7 root qpos -> 6 root DOFs)
    soft_lower_act = soft_lower[6:]
    soft_upper_act = soft_upper[6:]

    # Default actuated DOF positions (skip 7 root qpos for hinge joints)
    default_dof_pos_act = default_qpos[7:]  # (num_act_dofs,)

    # Gravity vector
    gravity_vec = jnp.array([0.0, 0.0, -1.0])

    # Commands scale
    commands_scale = jnp.array([
        cfg.obs_scale_lin_vel,
        cfg.obs_scale_lin_vel,
        cfg.obs_scale_ang_vel,
    ])

    # Number of feet
    num_feet = feet_body_indices.shape[0]

    # Height-based contact detection thresholds
    # Foot is "in contact" if its z-position is below this (meters)
    foot_contact_threshold = 0.05
    # Pelvis has "fallen" if z-position is below this (meters)
    pelvis_fall_threshold = 0.3

    # -----------------------------------------------------------------------
    # Reset (single env)
    # -----------------------------------------------------------------------

    def _reset_single(rng):
        """Reset a single environment. Returns (EnvState_single, obs)."""
        rng, rng_cmd, rng_qpos = jax.random.split(rng, 3)

        # Initial MJX data from default qpos with small noise
        qpos_noise = jax.random.uniform(rng_qpos, (nq,), minval=-0.01,
                                         maxval=0.01)
        qpos = default_qpos + qpos_noise
        # Keep root quaternion normalized
        root_quat = qpos[3:7]
        root_quat = root_quat / jnp.maximum(jnp.linalg.norm(root_quat), 1e-12)
        qpos = qpos.at[3:7].set(root_quat)

        qvel = jnp.zeros(nv)

        data = mjx.make_data(mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(mjx_model, data)

        # Sample initial commands
        cmd = _sample_commands(rng_cmd)

        state = EnvState(
            mjx_data=data,
            episode_step=jnp.int32(0),
            commands=cmd,
            prev_actions=jnp.zeros(num_act_dofs),
            last_actions=jnp.zeros(num_act_dofs),
            last_dof_vel=jnp.zeros(num_act_dofs),
            rng=rng,
        )
        obs = _compute_obs(state)
        return state, obs

    # -----------------------------------------------------------------------
    # Step (single env)
    # -----------------------------------------------------------------------

    def _step_single(state, action):
        """Step a single environment. Returns (new_state, obs, reward, done, info)."""
        action = jnp.clip(action, -1.0, 1.0)

        # Scale actions to target positions
        target = action * cfg.action_scale + default_dof_pos_act

        # Apply control
        data = state.mjx_data.replace(ctrl=target)

        # Physics step
        data = mjx.step(mjx_model, data)

        # Save last dof vel for acceleration reward
        last_dof_vel = state.mjx_data.qvel[6:]

        # Update state
        episode_step = state.episode_step + 1

        # Phase
        phase = (episode_step.astype(jnp.float32) * dt) % cfg.phase_period / cfg.phase_period
        phase_left = phase
        phase_right = (phase + cfg.phase_offset) % 1.0

        # Extract physics quantities
        root_quat = data.qpos[3:7]  # (w, x, y, z)
        root_pos = data.qpos[0:3]
        root_vel = data.qvel[0:3]   # linear velocity in world frame
        root_ang_vel = data.qvel[3:6]  # angular velocity in world frame
        dof_pos = data.qpos[7:]     # actuated joint positions
        dof_vel = data.qvel[6:]     # actuated joint velocities

        # Local frame quantities
        base_lin_vel = quat_rotate_inverse(root_quat, root_vel)
        base_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel)
        projected_gravity = quat_rotate_inverse(root_quat, gravity_vec)

        # Feet state
        feet_pos = data.xpos[feet_body_indices]     # (num_feet, 3)
        # Approximate feet velocity from body velocities
        # cvel is (nbody, 6) with [angular(3), linear(3)]
        feet_vel = data.cvel[feet_body_indices, 3:6]  # (num_feet, 3)

        # Contact detection via height proxy.
        # cfrc_ext is NOT computed by mjx.step() (requires rne_postconstraint),
        # so we use body z-position as a proxy for ground contact.
        feet_contact = feet_pos[:, 2] < foot_contact_threshold  # (num_feet,)

        # Pelvis contact (termination) — pelvis too low = fallen
        pelvis_z = data.xpos[pelvis_body_index, 2]
        pelvis_contact = pelvis_z < pelvis_fall_threshold

        # Termination
        timeout = episode_step >= max_episode_steps
        done = pelvis_contact | timeout

        # --- Rewards ---
        rewards = {}

        # 1. Tracking linear velocity
        lin_vel_error = jnp.sum(jnp.square(state.commands[:2] - base_lin_vel[:2]))
        rewards["tracking_lin_vel"] = jnp.exp(-lin_vel_error / cfg.tracking_sigma) * cfg.reward_tracking_lin_vel

        # 2. Tracking angular velocity
        ang_vel_error = jnp.square(state.commands[2] - base_ang_vel[2])
        rewards["tracking_ang_vel"] = jnp.exp(-ang_vel_error / cfg.tracking_sigma) * cfg.reward_tracking_ang_vel

        # 3. Linear velocity Z penalty
        rewards["lin_vel_z"] = jnp.square(base_lin_vel[2]) * cfg.reward_lin_vel_z

        # 4. Angular velocity XY penalty
        rewards["ang_vel_xy"] = jnp.sum(jnp.square(base_ang_vel[:2])) * cfg.reward_ang_vel_xy

        # 5. Orientation penalty
        rewards["orientation"] = jnp.sum(jnp.square(projected_gravity[:2])) * cfg.reward_orientation

        # 6. Base height penalty
        rewards["base_height"] = jnp.square(root_pos[2] - cfg.base_height_target) * cfg.reward_base_height

        # 7. DOF acceleration penalty
        dof_acc = (dof_vel - last_dof_vel) / dt
        rewards["dof_acc"] = jnp.sum(jnp.square(dof_acc)) * cfg.reward_dof_acc

        # 8. DOF velocity penalty
        rewards["dof_vel"] = jnp.sum(jnp.square(dof_vel)) * cfg.reward_dof_vel

        # 9. Action rate penalty
        rewards["action_rate"] = jnp.sum(jnp.square(state.prev_actions - action)) * cfg.reward_action_rate

        # 10. DOF position limits penalty
        below = jnp.clip(soft_lower_act - dof_pos, a_min=0.0)
        above = jnp.clip(dof_pos - soft_upper_act, a_min=0.0)
        rewards["dof_pos_limits"] = jnp.sum(below + above) * cfg.reward_dof_pos_limits

        # 11. Alive bonus
        rewards["alive"] = cfg.reward_alive

        # 12. Hip position penalty
        hip_pos_vals = dof_pos[hip_dof_indices - 6]  # offset by root DOFs
        rewards["hip_pos"] = jnp.sum(jnp.square(hip_pos_vals)) * cfg.reward_hip_pos

        # 13. Contact no velocity penalty
        contact_vel = feet_vel * feet_contact[:, None]  # zero out swing feet
        rewards["contact_no_vel"] = jnp.sum(jnp.square(contact_vel)) * cfg.reward_contact_no_vel

        # 14. Feet swing height penalty
        height_error = jnp.square(feet_pos[:, 2] - cfg.feet_swing_height_target)
        swing_mask = ~feet_contact  # penalize only swing feet
        rewards["feet_swing_height"] = jnp.sum(height_error * swing_mask) * cfg.reward_feet_swing_height

        # 15. Phase-aware contact reward
        leg_phase = jnp.array([phase_left, phase_right])
        is_stance = leg_phase < 0.55
        contact_match = ~(feet_contact ^ is_stance)
        rewards["contact"] = jnp.sum(contact_match.astype(jnp.float32)) * cfg.reward_contact

        total_reward = sum(rewards.values())

        # Command resampling
        rng, rng_cmd = jax.random.split(state.rng)
        should_resample = (episode_step % cmd_resample_steps) == 0
        new_cmd = jax.lax.cond(
            should_resample,
            lambda: _sample_commands(rng_cmd),
            lambda: state.commands,
        )

        # Auto-reset on done
        rng, rng_reset = jax.random.split(rng)
        reset_state, reset_obs = _reset_single(rng_reset)

        new_state = EnvState(
            mjx_data=jax.tree.map(
                lambda r, c: jnp.where(done, r, c),
                reset_state.mjx_data, data),
            episode_step=jnp.where(done, jnp.int32(0), episode_step),
            commands=jnp.where(done, reset_state.commands, new_cmd),
            prev_actions=jnp.where(done, jnp.zeros(num_act_dofs), action),
            last_actions=jnp.where(done, jnp.zeros(num_act_dofs), state.prev_actions),
            last_dof_vel=jnp.where(done, jnp.zeros(num_act_dofs), last_dof_vel),
            rng=rng,
        )

        obs = _compute_obs(new_state)

        info = {"reward_details": rewards, "timeout": timeout}
        return new_state, obs, total_reward, done, info

    # -----------------------------------------------------------------------
    # Observation computation
    # -----------------------------------------------------------------------

    def _compute_obs(state):
        """Compute observation vector for a single environment."""
        data = state.mjx_data
        root_quat = data.qpos[3:7]
        root_vel = data.qvel[0:3]
        root_ang_vel = data.qvel[3:6]
        dof_pos = data.qpos[7:]
        dof_vel = data.qvel[6:]

        base_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel)
        projected_gravity = quat_rotate_inverse(root_quat, gravity_vec)

        phase = (state.episode_step.astype(jnp.float32) * dt) % cfg.phase_period / cfg.phase_period

        obs = jnp.concatenate([
            base_ang_vel * cfg.obs_scale_ang_vel,                    # 3
            projected_gravity,                                        # 3
            state.commands * commands_scale,                          # 3
            (dof_pos - default_dof_pos_act) * cfg.obs_scale_dof_pos, # N
            dof_vel * cfg.obs_scale_dof_vel,                         # N
            state.prev_actions,                                       # N
            jnp.array([jnp.sin(2 * jnp.pi * phase)]),               # 1
            jnp.array([jnp.cos(2 * jnp.pi * phase)]),               # 1
        ])
        return obs

    # -----------------------------------------------------------------------
    # Command sampling
    # -----------------------------------------------------------------------

    def _sample_commands(rng):
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        cmd_x = jax.random.uniform(rng1, (), minval=cfg.cmd_x_vel_min,
                                    maxval=cfg.cmd_x_vel_max)
        cmd_y = jax.random.uniform(rng2, (), minval=cfg.cmd_y_vel_min,
                                    maxval=cfg.cmd_y_vel_max)
        cmd_yaw = jax.random.uniform(rng3, (), minval=cfg.cmd_yaw_vel_min,
                                      maxval=cfg.cmd_yaw_vel_max)
        return jnp.array([cmd_x, cmd_y, cmd_yaw])

    # -----------------------------------------------------------------------
    # Vectorized versions
    # -----------------------------------------------------------------------

    @jax.jit
    def reset(rng_batch):
        """Reset all environments. rng_batch: (num_envs, 2) PRNGKeys."""
        return jax.vmap(_reset_single)(rng_batch)

    @jax.jit
    def step(states, actions):
        """Step all environments.

        Args:
            states: EnvState with batched leaves (num_envs, ...)
            actions: (num_envs, num_act_dofs)

        Returns:
            (new_states, obs, rewards, dones, infos)
        """
        return jax.vmap(_step_single)(states, actions)

    return {
        "reset": reset,
        "step": step,
        "reset_single": _reset_single,
        "step_single": _step_single,
        "obs_dim": obs_dim,
        "act_dim": num_act_dofs,
        "num_act_dofs": num_act_dofs,
        "dt": dt,
        "max_episode_steps": max_episode_steps,
        "cfg": cfg,
    }
