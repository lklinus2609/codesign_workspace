# GBC (Gradient-Based Co-Design) Project: Rules and Conventions

## Project Overview
Gradient-Based Co-Design for the G1 humanoid robot. Uses orthogonal SPSA gradients
with a Cautious BFGS optimizer to jointly optimize morphology (joint oblique angles)
and locomotion policy (MimicKit AMP).

## Active Codebase
```
codesign/
├── __init__.py
├── codesign_g1_unified.py   # Main entry point (single-process, multi-GPU)
└── g1_mjcf_modifier.py      # Morphology parameterization (theta → MJCF quaternions)
```

## Critical Rules

### API Call Rules
- **NEVER guess API calls** - Always verify by reading source code in `newton/` and `MimicKit/`
- **NEVER assume parameter names** - Check function signatures in actual implementation files
- **ALWAYS refer to existing examples** before implementing new functionality:
  - Newton robot examples: `newton/newton/examples/robot/`
  - MimicKit training: `MimicKit/mimickit/run.py`
  - MimicKit AMP: `MimicKit/mimickit/learning/amp_agent.py`

### Newton Physics Engine
- Public API only from `newton/` module, **never import from `newton._src/`**
- Key classes:
  - `newton.ModelBuilder` - Build simulation models
  - `newton.solvers.SolverMuJoCo` - MuJoCo-based solver (used by MimicKit inner loop)
- Joint parameters (from `model.py`):
  - `joint_X_p`: Joint transform in parent frame [joint_count, 7] (pos + quat xyzw)
  - `joint_axis`: Joint axis in child frame [joint_dof_count, 3]
- Warp quaternion convention: **(x, y, z, w)** — identity `(0,0,0,1)`

### MimicKit Framework
- Agent hierarchy: `BaseAgent` -> `PPOAgent` -> `AMPAgent`
- Engine: `newton_engine.py` (sets `wp.config.enable_backward = False`)
- Configuration: `MimicKit/data/{agents,envs,engines}/`

### Design Parameters
- **6 symmetric lower-body joint pairs** (oblique angles, ±30 deg)
- **Bounds**: ±0.5236 radians
- **Symmetric pairs**: Left/right joints share parameter magnitude
- **Outer loop**: Orthogonal SPSA gradient → Cautious BFGS update
- **Inner loop**: MimicKit AMP with reward ramp (disc 1.0→0.1, task 0.0→0.9)

### Objective
- `reward = -CoT + vel_reward_weight * exp(-|v - v_cmd|^2 / sigma)`
- CoT = mechanical_power / (m * g * safe_v)

### Inner Loop Convergence Detection
- **Algorithm**: Slope-based linear regression on plateau values, NOT range-based.
  Range-based `(max-min)/|mean|` fires prematurely on smooth slow descent because
  the range can be small even when descent rate is non-zero. See `_check_slope_plateau`
  in `codesign_g1_unified.py`.
- **Trigger condition**: `|slope| * (window-1) / |mean| < threshold`.
- **Default window**: 30 outputs (raised from 10 after observing premature trigger
  at inner CoT 0.460 vs baseline asymptote 0.419 → ~9% policy headroom unexploited,
  envelope-theorem assumption broken).
- **Default thresholds**: `--inner-cot-plateau-threshold 0.0005`,
  `--task-plateau-threshold 0.02`.
- **Calibration**: monitor `gate/cot_slope_rel` in wandb. If it floors above the
  configured threshold even at long inner times, the slope estimator's noise floor
  is binding (~`σ·(W-1)/(σ_t·√W·|mean|)`) → relax threshold rather than tighten.

## Coding Conventions
- Classes: PascalCase (e.g., `CautiousBFGS`, `EpisodeCoTTracker`)
- Functions/methods: snake_case (e.g., `compute_spsa_gradient_parallel`)
- Constants: UPPER_SNAKE_CASE (e.g., `THETA_BOUNDS`, `GRAVITY`)
