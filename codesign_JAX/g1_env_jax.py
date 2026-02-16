"""
G1 Walking Environment in Pure JAX — IsaacLab-style rewards.

Vectorized via jax.vmap. All reward computation is pure JAX functions
(no Python control flow on data) for jax.jit + jax.vmap compatibility.

Observation:
    [ang_vel(3), projected_gravity(3), commands(3),
     dof_pos_relative(N), dof_vel(N), prev_actions(N),
     sin_phase(1), cos_phase(1)]

Rewards adapted from IsaacLab G1 flat terrain config:
    track_lin_vel_xy_exp, track_ang_vel_z_exp, feet_air_time,
    termination_penalty, flat_orientation_l2, lin_vel_z_l2,
    ang_vel_xy_l2, dof_acc_l2, action_rate_l2, feet_slide,
    dof_pos_limits, joint_deviation_hip, joint_deviation_arms,
    joint_deviation_torso

Termination: pelvis contact with ground OR episode timeout (20s)
"""

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
    episode_step: jnp.ndarray   # () int32
    commands: jnp.ndarray       # (3,)
    prev_actions: jnp.ndarray   # (num_act,)
    last_actions: jnp.ndarray   # (num_act,)
    last_dof_vel: jnp.ndarray   # (nv-6,)
    feet_air_time: jnp.ndarray  # (num_feet,) seconds in air
    feet_contact_time: jnp.ndarray  # (num_feet,) seconds in contact
    rng: jnp.ndarray            # (2,) PRNGKey


# ---------------------------------------------------------------------------
# Configuration — IsaacLab G1 flat terrain
# ---------------------------------------------------------------------------

class EnvConfig(NamedTuple):
    """Static environment configuration."""
    # Observation scales
    obs_scale_ang_vel: float = 0.25
    obs_scale_dof_pos: float = 1.0
    obs_scale_dof_vel: float = 0.05
    obs_scale_lin_vel: float = 2.0

    # Commands (IsaacLab G1 flat: forward + lateral + yaw)
    cmd_x_vel_min: float = 0.0
    cmd_x_vel_max: float = 1.0
    cmd_y_vel_min: float = -0.5
    cmd_y_vel_max: float = 0.5
    cmd_yaw_vel_min: float = -1.0
    cmd_yaw_vel_max: float = 1.0
    cmd_resample_time: float = 10.0  # seconds

    # Control
    action_scale: float = 0.25

    # Reward scales (IsaacLab G1 flat terrain)
    reward_track_lin_vel: float = 1.0
    reward_track_ang_vel: float = 1.0
    reward_feet_air_time: float = 0.75
    reward_termination: float = -200.0
    reward_orientation: float = -1.0
    reward_lin_vel_z: float = -0.2
    reward_ang_vel_xy: float = -0.05
    reward_dof_acc: float = -1.0e-7
    reward_action_rate: float = -0.005
    reward_feet_slide: float = -0.1
    reward_dof_pos_limits: float = -1.0
    reward_joint_deviation_hip: float = -0.1
    reward_joint_deviation_arms: float = -0.1
    reward_joint_deviation_torso: float = -0.1

    # Reward parameters
    tracking_sigma: float = 0.25
    soft_dof_pos_limit: float = 0.9
    feet_air_time_threshold: float = 0.4  # max rewarded air time (seconds)

    # Termination
    max_episode_length_s: float = 20.0

    # Phase (gait timing)
    phase_period: float = 0.8
    phase_offset: float = 0.5


# ---------------------------------------------------------------------------
# Quaternion utilities (MuJoCo wxyz convention)
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (wxyz)."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = jnp.cross(q_vec, v) * q_w * 2.0
    c = q_vec * jnp.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def yaw_quat(q):
    """Extract yaw-only quaternion from full quaternion (wxyz)."""
    # q = (w, x, y, z) -> yaw = atan2(2(wz+xy), 1-2(y²+z²))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    half_yaw = yaw / 2.0
    return jnp.stack([jnp.cos(half_yaw), jnp.zeros_like(half_yaw),
                      jnp.zeros_like(half_yaw), jnp.sin(half_yaw)], axis=-1)


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
    total_mass = metadata["total_mass"]

    # Joint group indices for deviation rewards
    hip_dof_indices = metadata["hip_dof_indices"]
    torso_dof_indices = metadata["torso_dof_indices"]
    arm_dof_indices = metadata["arm_dof_indices"]
    leg_dof_indices = metadata["leg_dof_indices"]

    # Actuated DOFs start after the 6 root DOFs (freejoint)
    num_act_dofs = nv - 6
    obs_dim = 3 + 3 + 3 + 3 * num_act_dofs + 2

    max_episode_steps = int(cfg.max_episode_length_s / dt)
    cmd_resample_steps = max(1, int(cfg.cmd_resample_time / dt))

    # Soft DOF limits (for ankle joints, matching IsaacLab)
    mid = (dof_pos_upper + dof_pos_lower) / 2
    half_range = (dof_pos_upper - dof_pos_lower) / 2
    soft_lower = mid - cfg.soft_dof_pos_limit * half_range
    soft_upper = mid + cfg.soft_dof_pos_limit * half_range
    # Only actuated DOFs (skip 6 root DOFs)
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
    foot_contact_threshold = 0.05
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
            feet_air_time=jnp.zeros(num_feet),
            feet_contact_time=jnp.zeros(num_feet),
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

        # Extract physics quantities
        root_quat = data.qpos[3:7]  # (w, x, y, z)
        root_pos = data.qpos[0:3]
        root_vel = data.qvel[0:3]   # linear velocity in world frame
        root_ang_vel = data.qvel[3:6]  # angular velocity in world frame
        dof_pos = data.qpos[7:]     # actuated joint positions
        dof_vel = data.qvel[6:]     # actuated joint velocities

        # Yaw-frame velocity (IsaacLab style)
        yaw_q = yaw_quat(root_quat)
        base_lin_vel_yaw = quat_rotate_inverse(yaw_q, root_vel)

        # Body-frame quantities
        base_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel)
        projected_gravity = quat_rotate_inverse(root_quat, gravity_vec)

        # Feet state
        feet_pos = data.xpos[feet_body_indices]     # (num_feet, 3)
        # cvel is (nbody, 6) with [angular(3), linear(3)]
        feet_vel = data.cvel[feet_body_indices, 3:6]  # (num_feet, 3)

        # Contact detection via height proxy
        feet_contact = feet_pos[:, 2] < foot_contact_threshold  # (num_feet,)

        # Update air/contact time tracking
        new_air_time = jnp.where(
            feet_contact,
            jnp.zeros(num_feet),                    # reset air time on contact
            state.feet_air_time + dt,                # increment air time
        )
        new_contact_time = jnp.where(
            feet_contact,
            state.feet_contact_time + dt,            # increment contact time
            jnp.zeros(num_feet),                     # reset contact time on air
        )

        # Pelvis contact (termination) — pelvis too low = fallen
        pelvis_z = data.xpos[pelvis_body_index, 2]
        pelvis_contact = pelvis_z < pelvis_fall_threshold

        # Termination
        timeout = episode_step >= max_episode_steps
        terminated = pelvis_contact  # true termination (not timeout)
        done = terminated | timeout

        # --- Rewards (IsaacLab G1 flat terrain) ---
        rewards = {}

        # 1. Track linear velocity XY (yaw-frame, exponential)
        lin_vel_error = jnp.sum(jnp.square(
            state.commands[:2] - base_lin_vel_yaw[:2]))
        rewards["track_lin_vel"] = (
            jnp.exp(-lin_vel_error / cfg.tracking_sigma)
            * cfg.reward_track_lin_vel)

        # 2. Track angular velocity Z (world frame, exponential)
        ang_vel_error = jnp.square(state.commands[2] - root_ang_vel[2])
        rewards["track_ang_vel"] = (
            jnp.exp(-ang_vel_error / cfg.tracking_sigma)
            * cfg.reward_track_ang_vel)

        # 3. Feet air time (biped single-stance reward)
        # Reward when exactly 1 foot is in contact (single stance phase)
        in_contact = new_contact_time > 0.0
        num_in_contact = jnp.sum(in_contact.astype(jnp.int32))
        single_stance = (num_in_contact == 1)
        # For each foot, take the current mode time (air or contact)
        mode_time = jnp.where(in_contact, new_contact_time, new_air_time)
        # In single stance: reward = min mode time across feet, clamped
        air_time_reward = jnp.where(
            single_stance,
            jnp.clip(jnp.min(mode_time), 0.0, cfg.feet_air_time_threshold),
            0.0,
        )
        # Only reward when moving (command velocity > threshold)
        cmd_vel_norm = jnp.linalg.norm(state.commands[:2])
        air_time_reward = air_time_reward * (cmd_vel_norm > 0.1)
        rewards["feet_air_time"] = air_time_reward * cfg.reward_feet_air_time

        # 4. Termination penalty
        rewards["termination"] = terminated.astype(jnp.float32) * cfg.reward_termination

        # 5. Flat orientation (projected gravity xy should be zero)
        rewards["orientation"] = (
            jnp.sum(jnp.square(projected_gravity[:2]))
            * cfg.reward_orientation)

        # 6. Linear velocity Z penalty
        rewards["lin_vel_z"] = (
            jnp.square(base_lin_vel_yaw[2])
            * cfg.reward_lin_vel_z)

        # 7. Angular velocity XY penalty
        rewards["ang_vel_xy"] = (
            jnp.sum(jnp.square(base_ang_vel[:2]))
            * cfg.reward_ang_vel_xy)

        # 8. DOF acceleration penalty (hip + knee joints only)
        dof_acc = (dof_vel - last_dof_vel) / dt
        leg_acc = dof_acc[leg_dof_indices - 6]  # offset by root DOFs
        rewards["dof_acc"] = (
            jnp.sum(jnp.square(leg_acc))
            * cfg.reward_dof_acc)

        # 9. Action rate penalty
        rewards["action_rate"] = (
            jnp.sum(jnp.square(state.prev_actions - action))
            * cfg.reward_action_rate)

        # 10. Feet slide penalty (foot velocity during contact)
        feet_vel_xy = feet_vel[:, :2]  # (num_feet, 2)
        slide_speed = jnp.linalg.norm(feet_vel_xy, axis=-1)  # (num_feet,)
        rewards["feet_slide"] = (
            jnp.sum(slide_speed * feet_contact.astype(jnp.float32))
            * cfg.reward_feet_slide)

        # 11. DOF position limits penalty (ankle joints only)
        ankle_idx = metadata["ankle_dof_indices"] - 6
        ankle_pos = dof_pos[ankle_idx]
        ankle_soft_lower = soft_lower_act[ankle_idx]
        ankle_soft_upper = soft_upper_act[ankle_idx]
        below = jnp.clip(ankle_soft_lower - ankle_pos, a_min=0.0)
        above = jnp.clip(ankle_pos - ankle_soft_upper, a_min=0.0)
        rewards["dof_pos_limits"] = (
            jnp.sum(below + above)
            * cfg.reward_dof_pos_limits)

        # 12. Joint deviation: hip (hip_yaw + hip_roll, L1)
        hip_idx = hip_dof_indices - 6
        hip_default = default_dof_pos_act[hip_idx]
        rewards["joint_dev_hip"] = (
            jnp.sum(jnp.abs(dof_pos[hip_idx] - hip_default))
            * cfg.reward_joint_deviation_hip)

        # 13. Joint deviation: arms (shoulder + elbow + wrist, L1)
        arm_idx = arm_dof_indices - 6
        arm_default = default_dof_pos_act[arm_idx]
        rewards["joint_dev_arms"] = (
            jnp.sum(jnp.abs(dof_pos[arm_idx] - arm_default))
            * cfg.reward_joint_deviation_arms)

        # 14. Joint deviation: torso (waist joints, L1)
        torso_idx = torso_dof_indices - 6
        torso_default = default_dof_pos_act[torso_idx]
        rewards["joint_dev_torso"] = (
            jnp.sum(jnp.abs(dof_pos[torso_idx] - torso_default))
            * cfg.reward_joint_deviation_torso)

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
            feet_air_time=jnp.where(done, jnp.zeros(num_feet), new_air_time),
            feet_contact_time=jnp.where(done, jnp.zeros(num_feet), new_contact_time),
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
        """Step all environments."""
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
