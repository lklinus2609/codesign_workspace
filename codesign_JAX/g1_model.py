"""
MJX Model Loading for G1 Humanoid.

Loads the G1 MJCF, configures Newton solver for differentiable simulation,
converts to MJX, and extracts body/joint/actuator metadata.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent
DEFAULT_MJCF_PATH = _PROJECT_ROOT / "MimicKit" / "data" / "assets" / "g1" / "g1.xml"

# 6 symmetric pairs of body names whose frame quaternions we parameterize
SYMMETRIC_PAIRS = [
    ("left_hip_pitch_link", "right_hip_pitch_link"),
    ("left_hip_roll_link", "right_hip_roll_link"),
    ("left_hip_yaw_link", "right_hip_yaw_link"),
    ("left_knee_link", "right_knee_link"),
    ("left_ankle_pitch_link", "right_ankle_pitch_link"),
    ("left_ankle_roll_link", "right_ankle_roll_link"),
]
NUM_DESIGN_PARAMS = len(SYMMETRIC_PAIRS)  # 6


def load_g1_model(mjcf_path=None):
    """Load G1 MuJoCo model and configure for MJX differentiable simulation.

    Args:
        mjcf_path: Path to g1.xml. Defaults to MimicKit asset.

    Returns:
        mj_model: mujoco.MjModel (CPU)
        mjx_model: mjx.Model (JAX, on default device)
        metadata: dict with body/joint/actuator info
    """
    if mjcf_path is None:
        mjcf_path = str(DEFAULT_MJCF_PATH)

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    # Configure Newton solver (CG uses jax.lax.while_loop with dynamic
    # termination which blocks reverse-mode differentiation)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 1
    mj_model.opt.ls_iterations = 4

    # Convert to MJX
    mjx_model = mjx.put_model(mj_model)

    # Extract metadata
    metadata = _extract_metadata(mj_model)

    print(f"[G1Model] Loaded: {mjcf_path}")
    print(f"  Bodies: {mj_model.nbody}, Joints: {mj_model.njnt}, "
          f"DOFs: {mj_model.nv}, Actuators: {mj_model.nu}")
    print(f"  Solver: Newton (iters={mj_model.opt.iterations}, "
          f"ls_iters={mj_model.opt.ls_iterations})")
    print(f"  Timestep: {mj_model.opt.timestep}s")

    return mj_model, mjx_model, metadata


def _extract_metadata(mj_model):
    """Extract body indices, joint info, actuator mapping from MjModel."""
    metadata = {}

    # Body name -> index mapping
    body_names = {}
    for i in range(mj_model.nbody):
        name = mj_model.body(i).name
        if name:
            body_names[name] = i
    metadata["body_names"] = body_names

    # Joint name -> index mapping
    joint_names = {}
    for i in range(mj_model.njnt):
        name = mj_model.jnt(i).name
        if name:
            joint_names[name] = i
    metadata["joint_names"] = joint_names

    # Actuator name -> index mapping
    actuator_names = {}
    for i in range(mj_model.nu):
        name = mj_model.actuator(i).name
        if name:
            actuator_names[name] = i
    metadata["actuator_names"] = actuator_names

    # Parameterized body indices (for morphology optimization)
    param_body_indices = []
    for left_name, right_name in SYMMETRIC_PAIRS:
        left_idx = body_names.get(left_name)
        right_idx = body_names.get(right_name)
        if left_idx is None or right_idx is None:
            raise ValueError(f"Body not found: {left_name} or {right_name}")
        param_body_indices.append((left_idx, right_idx))
    metadata["param_body_indices"] = param_body_indices

    # Flat list of all parameterized body indices (12 total = 6 pairs)
    all_param_indices = []
    param_for_body = []  # which theta index each body maps to
    for i, (left_idx, right_idx) in enumerate(param_body_indices):
        all_param_indices.extend([left_idx, right_idx])
        param_for_body.extend([i, i])
    metadata["all_param_body_indices"] = jnp.array(all_param_indices, dtype=jnp.int32)
    metadata["param_for_body"] = jnp.array(param_for_body, dtype=jnp.int32)

    # Feet body indices (for contact rewards)
    # Explicitly order: left first, right second (matches phase assignment)
    left_foot_idx = body_names["left_ankle_roll_link"]
    right_foot_idx = body_names["right_ankle_roll_link"]
    metadata["feet_body_indices"] = jnp.array([left_foot_idx, right_foot_idx],
                                               dtype=jnp.int32)
    metadata["feet_names"] = ["left_ankle_roll_link", "right_ankle_roll_link"]

    # Pelvis body index (for termination)
    metadata["pelvis_body_index"] = body_names.get("pelvis", 0)

    # Hip DOF indices (for hip_pos reward)
    hip_dof_indices = []
    for jname, jidx in joint_names.items():
        if "hip" in jname and "pitch" not in jname:
            # Get DOF address for this joint
            dof_adr = mj_model.jnt_dofadr[jidx]
            hip_dof_indices.append(dof_adr)
    metadata["hip_dof_indices"] = jnp.array(hip_dof_indices, dtype=jnp.int32)

    # Default qpos (for reset and relative observations)
    metadata["default_qpos"] = jnp.array(mj_model.qpos0.copy())

    # DOF limits
    # jnt_range is (njnt, 2) - but we need per-DOF limits
    # For hinge joints (all G1 actuated joints are hinge): 1 DOF per joint
    dof_lower = np.zeros(mj_model.nv)
    dof_upper = np.zeros(mj_model.nv)
    for j in range(mj_model.njnt):
        jtype = mj_model.jnt_type[j]
        dof_adr = mj_model.jnt_dofadr[j]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            # Free joint: 6 DOFs, no limits
            dof_lower[dof_adr:dof_adr+6] = -np.inf
            dof_upper[dof_adr:dof_adr+6] = np.inf
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
            # jnt_range is already in radians after MuJoCo compilation
            rng = mj_model.jnt_range[j]
            dof_lower[dof_adr] = rng[0] if mj_model.jnt_limited[j] else -np.inf
            dof_upper[dof_adr] = rng[1] if mj_model.jnt_limited[j] else np.inf
    metadata["dof_pos_lower"] = jnp.array(dof_lower)
    metadata["dof_pos_upper"] = jnp.array(dof_upper)

    # Number of actuated DOFs (excludes 6 root DOFs from freejoint)
    metadata["num_actuators"] = mj_model.nu
    metadata["nv"] = mj_model.nv
    metadata["nq"] = mj_model.nq
    metadata["nbody"] = mj_model.nbody
    metadata["timestep"] = mj_model.opt.timestep

    # Sanity check: for G1 all actuated joints are hinge, so nu == nv - 6
    assert mj_model.nu == mj_model.nv - 6, (
        f"Expected nu={mj_model.nu} == nv-6={mj_model.nv - 6}. "
        f"Model has unactuated joints or non-hinge joints."
    )

    # Geom-to-body mapping (for contact detection)
    # Build list of geom IDs belonging to feet and pelvis
    feet_geom_ids = []
    pelvis_geom_ids = []
    for g in range(mj_model.ngeom):
        body_id = mj_model.geom_bodyid[g]
        geom_name = mj_model.geom(g).name
        if body_id == left_foot_idx or body_id == right_foot_idx:
            feet_geom_ids.append(g)
        if body_id == body_names.get("pelvis", -1):
            pelvis_geom_ids.append(g)
    metadata["feet_geom_ids"] = set(feet_geom_ids)
    metadata["pelvis_geom_ids"] = set(pelvis_geom_ids)
    # Floor geom is body 0 ("world"), geom index 0 (the ground plane)
    # In MuJoCo, the ground plane geom is typically the first geom of worldbody
    floor_geom_ids = []
    for g in range(mj_model.ngeom):
        if mj_model.geom_bodyid[g] == 0:
            floor_geom_ids.append(g)
    metadata["floor_geom_ids"] = set(floor_geom_ids)

    # Total mass
    metadata["total_mass"] = float(np.sum(mj_model.body_mass))

    # Base body quaternions (for morphology - before any theta modification)
    # MuJoCo body_quat is (nbody, 4) in (w,x,y,z) format
    metadata["base_body_quat"] = jnp.array(mj_model.body_quat.copy())

    return metadata


def make_mjx_data(mj_model, mjx_model):
    """Create initial MJX data state from model defaults.

    Returns:
        mjx_data: mjx.Data initialized to default qpos
    """
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    mjx_data = mjx.put_data(mj_model, mj_data)
    return mjx_data
