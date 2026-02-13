# PGHC Co-Design Project: Rules and Conventions

## Project Overview
Implementing Algorithm 1: **Performance-Gated Hybrid Co-Design (PGHC)** from the Masters Thesis for the G1 humanoid robot.

**Key Features (per Thesis):**
1. **Stability Gating**: Dynamic policy convergence detection via delta_rel metric
2. **Adaptive Trust Region**: Performance validation before design updates
3. **Multi-DOF Morphology**: Support for 6-30 joint oblique angles
4. **Cost of Transport Objective**: Energy efficiency optimization

## Critical Rules

### 0. Development Process Rules
- **NEVER leave placeholder functions** without documenting them in IMPLEMENTATION_PLAN.md with reason and timeline
- **NEVER make workarounds or change plans** without explicit user confirmation
- **ALWAYS update documentation** (this file and IMPLEMENTATION_PLAN.md) when milestones are achieved
- **ALWAYS record** any deviations from Algorithm 1 with justification and user approval

### 1. API Call Rules
- **NEVER guess API calls** - Always verify by reading source code in `newton/` and `MimicKit/`
- **NEVER assume parameter names** - Check function signatures in actual implementation files
- **ALWAYS refer to existing examples** before implementing new functionality:
  - Newton diffsim examples: `newton/newton/examples/diffsim/`
  - Newton robot examples: `newton/newton/examples/robot/`
  - MimicKit training: `MimicKit/mimickit/run.py`
  - MimicKit AMP: `MimicKit/mimickit/learning/amp_agent.py`

### 2. Newton Physics Engine
- Public API only from `newton/` module, **never import from `newton._src/`**
- Key classes:
  - `newton.ModelBuilder` - Build simulation models
  - `newton.solvers.SolverMuJoCo` - MuJoCo-based solver (used by MimicKit)
  - `newton.solvers.SolverSemiImplicit` - Differentiable solver for BPTT
- Differentiable simulation requires:
  - `model = builder.finalize(requires_grad=True)`
  - `wp.Tape()` for recording forward pass
  - `tape.backward(loss)` for gradients
- Joint parameters (from `model.py`):
  - `joint_X_p`: Joint transform in parent frame [joint_count, 7] (pos + quat)
  - `joint_axis`: Joint axis in child frame [joint_dof_count, 3]

### 3. MimicKit Framework
- Entry point: `MimicKit/mimickit/run.py`
- Agent hierarchy: `BaseAgent` -> `PPOAgent` -> `AMPAgent`
- Engine abstraction: `MimicKit/mimickit/engines/`
  - `newton_engine.py` - Newton physics backend
  - `isaac_gym_engine.py` - Isaac Gym backend (default for training)
- Configuration files:
  - Agent configs: `MimicKit/data/agents/`
  - Environment configs: `MimicKit/data/envs/`
  - Engine configs: `MimicKit/data/engines/`

### 4. G1 Robot Model
- MuJoCo XML: `MimicKit/data/assets/g1/g1.xml`
- USD (Newton native): `newton/examples/assets/unitree_g1/usd/g1_isaac.usd`
- Hip structure (left side):
  ```
  pelvis
  └── left_hip_pitch_link (pos="0 0.064452 -0.1027")
      └── left_hip_roll_link (pos="0 0.052 -0.030465", quat="0.996179 0 -0.0873386 0")
          └── left_hip_yaw_link
              └── left_knee_link
  ```
- **Design parameter**: The quaternion on `left/right_hip_roll_link` bodies
  - Current: ~10 degree X-axis rotation built into kinematic chain
  - Target: Parameterize ±10 degrees from neutral

## Algorithm Parameters (PGHC - Algorithm 1)

### Phase 1: Performance-Gated Inner Loop (Lines 12-18)
- **Stability Window (W)**: 100 episodes for moving average
- **Convergence Threshold (delta_conv)**: 0.05 (5% relative change)
- **Stability Metric**: delta_rel = |R_t - R_{t-W}| / (|R_{t-W}| + epsilon)
- Outer loop triggers ONLY when delta_rel < delta_conv
- This enforces Envelope Theorem approximation validity

### Phase 2: Trust-Region Outer Loop (Lines 24-35)
- **Trust Region Threshold (xi)**: 0.1 (10% performance degradation allowed)
- **Adaptive Learning Rate (beta)**:
  - Initial: 0.01
  - Decays by 0.5x on trust region violation
  - Grows by 1.5x on small improvement
  - Bounds: [1e-5, 0.1]
- **Performance Check**: D = J(phi_k, theta*) - J(phi', theta*)
- Accept only if D > -xi * |J(phi_k)|

### Design Parameters (Morphology)
- **Lower Body (15 DOF)**: Hip, knee, ankle oblique angles
- **Full Body (30 DOF)**: All joint oblique angles
- **Bounds**: ±30 degrees (±0.5236 radians)
- **Symmetric Pairs**: Left/right joints share parameter magnitude
- **Update**: phi_{k+1} = proj_C(phi_k - beta * grad_phi L)

### Objective Functions
- **Cost of Transport (CoT)**: E_total / (m * g * d)
- **Upright Bonus**: Penalty for orientation deviation
- **Velocity Tracking**: Match target forward velocity

### Gating Modes
- **"stability"**: PGHC algorithm (recommended)
- **"fixed"**: Legacy warmup + frequency schedule (fallback)

## Coding Conventions

### File Organization
```
codesign/
├── hybrid_codesign/           # New module for co-design
│   ├── __init__.py
│   ├── hybrid_agent.py        # HybridAMPAgent class
│   ├── parametric_model.py    # Parametric G1 with differentiable morphology
│   ├── diff_rollout.py        # Differentiable rollout for outer loop
│   └── config/
│       ├── hybrid_agent.yaml
│       └── hybrid_env.yaml
├── CODESIGN_RULES.md          # This file
└── algorithm1.pdf             # Reference algorithm
```

### Naming Conventions
- Classes: PascalCase (e.g., `HybridAMPAgent`, `ParametricG1Model`)
- Functions/methods: snake_case (e.g., `compute_design_gradient`)
- Constants: UPPER_SNAKE_CASE (e.g., `DESIGN_PARAM_BOUNDS`)
- Config keys: snake_case (e.g., `outer_loop_freq`)

### Documentation
- All new classes/functions must have docstrings
- Reference Algorithm 1 line numbers in comments where applicable
- Log all design parameter changes and gradients

## Key Technical Constraints

### Newton Engine Limitations
1. Current `newton_engine.py` sets `requires_grad=False` (line 370)
   - Must modify to support differentiable mode for outer loop
2. MuJoCo solver may not support full BPTT
   - Consider using `SolverSemiImplicit` for outer loop rollouts
3. Joint transforms are set at finalization
   - Need mechanism to update `joint_X_p` tensor without rebuilding

### MimicKit Integration
1. AMP agent expects non-differentiable physics
   - Outer loop uses separate differentiable rollout
2. Policy must be frozen during outer loop gradient computation
   - Use `torch.no_grad()` for policy, only `joint_X_p` has gradients

## Testing Requirements
- Unit tests for parametric model (angle to transform conversion)
- Gradient verification via finite differences (see `example_diffsim_spring_cage.py`)
- Validate that inner loop (AMP) converges before outer loop activation
- Log and plot design parameter evolution

## References
- Algorithm 1: `codesign/algorithm1.pdf`
- Newton diffsim examples: `newton/newton/examples/diffsim/`
- MimicKit AMP implementation: `MimicKit/mimickit/learning/amp_agent.py`
- G1 robot model: `MimicKit/data/assets/g1/g1.xml`
