# AMP Retargeting for Co-Design — Implementation Plan

**Status**: Ready to implement. Start with the sanity check (§3), then Option A (§4). Fall back to Option B (§5) only if A fails empirical validation.

---

## 1. Problem

The codesign outer loop plateaus on CoT after ~30k inner iters / 12h. Root cause: the AMP discriminator's style reward has a **systematic, morphology-dependent bias** that injects non-behavioral noise into the SPSA gradient.

Empirical observations so far:
- **Position 1 (`w_disc=0` post-kickoff)**: gait corrupts — robot "drags feet" to minimize CoT. Unusable.
- **Current code (`w_disc=0.5` post-kickoff)**: gait stays natural but CoT plateaus. Outer loop can't find lower-CoT morphologies.

The dilemma: we need the style reward to anchor gait quality, but the style reward itself drifts with morphology changes, biasing the outer gradient.

---

## 2. Root Cause (from AMP code trace)

AMP discriminator receives two streams. **Policy** body positions come from the physics engine (which uses the *modified* `joint_X_p`). **Mocap** body positions come from `kin_char_model.forward_kinematics()` using `_local_rotation` / `_local_translation` loaded once at MJCF init and **never updated** when θ changes.

**Key files/lines**:
- `MimicKit/mimickit/envs/amp_env.py:76` — mocap FK call (`self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)`)
- `MimicKit/mimickit/envs/amp_env.py:169` — policy `body_pos = self._engine.get_body_pos(char_id)` (reads from modified physics)
- `MimicKit/mimickit/anim/kin_char_model.py:176-208` — `forward_kinematics()` uses `self._local_rotation[j]` (line 187) — this is the per-joint frame quat relative to parent. **This is what needs updating.**
- `MimicKit/mimickit/envs/amp_env.py:107` — `joint_rot = self._kin_char_model.dof_to_rot(dof_pos)` for policy; converts DOF→quat via joint axes that also rotate under morphology change (secondary bias).
- `codesign/g1_mjcf_modifier.py:110-118` — where we apply `δq = quat_from_x_rotation(θ)` to `joint_X_p` equivalent (MJCF body `quat` attribute).

Consequence: for a 0.05 rad perturbation, a ~1m leg's key-body position shifts by ~5 cm in world frame purely from the FK mismatch, independent of policy behavior. The discriminator sees a "worse" motion; `log D` drops; SPSA reads this as a gradient signal; outer loop gets pushed in the wrong direction.

---

## 3. Sanity Check (RUN FIRST — 30 min)

Before any implementation, validate the hypothesis is the actual cause of the plateau. If the bias is small relative to the expected CoT signal, we're chasing the wrong thing.

**Procedure**:
1. Load the kickoff-trained checkpoint from the 12h run.
2. For each of the 6 design params, sweep θ_i ∈ {θ₀ - ε, θ₀, θ₀ + ε} with ε = 0.05 rad (others fixed at θ₀).
3. For each θ: rollout frozen policy for 300 steps × N worlds, record per-step `log D(s,a)` from the discriminator.
4. Report: mean `log D` at each θ, and the directional derivative `(log D(θ+ε) - log D(θ-ε)) / (2ε)` per param.

**Decision rule**:
- If `|Δ log D|` per perturbation is ≳ 5× the typical per-step fluctuation of `log D` during training → bias is material, proceed with Option A.
- If `|Δ log D|` is comparable to training-time noise → bias is small, plateau is from a different cause (revisit trust region, SPSA SNR, or reward-weight mismatch from the previous review).

**Where to implement**: new script `codesign/diagnose_amp_morphology_bias.py` — imports MimicKit agent, loads checkpoint, wraps the SPSA rollout loop, dumps `log D` to a CSV + matplotlib plot.

---

## 4. Option A — Morphology-Aware Mocap Retargeting (PRIMARY)

**Idea**: when θ changes, update `kin_char_model._local_rotation` on the mocap side so that mocap FK produces body positions consistent with the *current* morphology. Discriminator then compares like-to-like. Mocap `joint_rot` and `_local_translation` stay unchanged — only the joint frame orientations (`_local_rotation`) shift to match `joint_X_p`.

### 4.1 What changes conceptually

For each of the 6 parameterized body pairs (e.g., `left_hip_pitch_link`), the MJCF modifier computes:
```
new_quat = quat_from_x_rotation(θ_i) * base_quat
```
and writes this to the body's MJCF `quat` attribute. The physics engine picks this up via `joint_X_p`. **The fix is to apply the same delta to `kin_char_model._local_rotation[j]`** for the matching joint index `j`.

### 4.2 Implementation steps

1. **Build a body-name → kin_char joint index map.**
   - `kin_char_model._joints` is a list of `Joint` objects loaded from MJCF.
   - Either: (a) add a `name_to_idx` dict to `KinCharModel`, or (b) iterate the joint list in the codesign script and match by name.
   - Prefer (a) — clean one-time addition to MimicKit.

2. **Cache original `_local_rotation` on env build.**
   - In `codesign_g1_unified.py`, after env construction, snapshot `env._kin_char_model._local_rotation.clone()` into an "original" buffer. We'll multiply `δq(θ) * original` at each θ update, never accumulate drift.

3. **Add `update_kin_char_local_rotations(env, theta)`.**
   - New helper in `codesign_g1_unified.py` (or a new `amp_retarget.py` module).
   - For each `(body_name, θ_i)`:
     - Look up joint index `j` via `name_to_idx`.
     - Compute `δq_i = quat_from_x_rotation(θ_i)` (already exists in `g1_mjcf_modifier.quat_from_x_rotation`, just convert to torch).
     - `env._kin_char_model._local_rotation[j] = quat_mul(δq_i, original_local_rotation[j])`.
   - Quaternion convention: MJCF uses `(w,x,y,z)`, Newton/physics may use `(x,y,z,w)`. Check `kin_char_model._local_rotation` convention (likely `(x,y,z,w)` per `torch_util.quat_mul`). Match conventions carefully.

4. **Call retargeting at every θ update.**
   - After `update_training_joint_X_p()` in `codesign_g1_unified.py` (currently at ~:1822), also call `update_kin_char_local_rotations(env, theta)`.
   - Also call during SPSA perturbation evaluation: `apply_partitioned_morphologies()` at :833-887 writes different `joint_X_p` slices per world. For mocap retargeting, we have a choice:
     - **Option A1 (cheap, biased-within-SPSA)**: retarget only to θ_center during SPSA. The per-perturbation bias still exists but is now smaller (centered on θ_center not θ₀).
     - **Option A2 (correct, more expensive)**: per-perturbation retargeting — each of the 37 world-groups gets its own mocap FK. Requires either (i) batching FK over the 37 morphologies (vectorized if `_local_rotation` can be a per-world tensor — likely not without refactor), or (ii) running the SPSA seed loop per-morphology serially (slow).
   - **Start with A1**. If A1 still plateaus, move to A2. A1 handles the outer-iter drift (largest source of bias); A2 handles per-perturbation drift (smaller but present).

5. **Verify `_local_rotation` is actually read per step.**
   - `kin_char_model.forward_kinematics()` line 187 reads `self._local_rotation[j]` every call. Good — mutation propagates without cache invalidation.
   - `dof_to_rot()` (kin_char_model.py:146) uses `self.axis` per joint (line 56 in `Joint.dof_to_rot`). If we change `_local_rotation`, does `axis` also need to change? Yes — the joint axis is in the *joint's frame*, which we just rotated. Decide: either rotate `axis` too, or leave it (the DOF value is a scalar around the axis, so rotating the frame rotates the effective axis of the resulting quat via `local_rot` composition in FK). **Think through this carefully before implementing** — test with a single param perturbation that both `joint_rot` and `body_pos` on the mocap side match what the physics engine produces for the same DOF trajectory.

### 4.3 Validation

**Test 1 — bias elimination (sanity check, re-run §3)**
- After Option A, re-run the sweep from §3.
- Expected: `|Δ log D| per ε` drops by 5-10× vs. the baseline measurement. Not zero (policy-side `joint_rot` via `dof_to_rot` still has secondary bias), but should be small enough not to dominate the SPSA gradient.

**Test 2 — mocap FK consistency**
- At θ = θ₀, mocap body positions (from `kin_char_model.FK`) should match the physics-engine body positions for a given DOF trajectory, up to a fixed offset (root frame).
- At θ = θ₀ + 0.1 rad (large perturbation), same equality should hold after retargeting.
- Write a `test_retargeting_consistency.py` that asserts max position diff < 1 mm across 100 random DOF samples.

**Test 3 — full training run**
- 12h run with Option A enabled, same hyperparameters as the previous 30k-step run.
- Primary metric: CoT trajectory over outer iterations. Expected: monotonic decrease (or at least slower plateau).
- Secondary: `envelope_grad_norm` (already logged at :1677) stays small post-ramp. Gait quality (visual inspection of recorded videos) stays natural.

### 4.4 Risks / open questions

- **Quaternion convention mismatches** between MJCF (`w,x,y,z`) and torch-util (`x,y,z,w`) are the most likely footgun. Write a small quaternion-convention test as part of test #2.
- **`Joint.axis`** in `kin_char_model.py:19` — whether this also needs rotation when we rotate `_local_rotation`. Best empirical test: compare policy-side `joint_rot` to mocap-side `joint_rot` for identical DOF values under perturbed morphology.
- **SPSA per-perturbation retargeting cost (A2)** — if needed, may require batching FK across morphologies, which is a MimicKit-internal refactor. Budget 1-2 days if we end up needing it.
- **Motion library caching** — `amp_env._fetch_disc_demo_data()` calls `_motion_lib.calc_motion_frame()` (amp_env.py:74), which returns `joint_rot` (stored raw, morphology-invariant) but NOT body_pos — body_pos is computed fresh each call via FK (line 76). Good, no cache to invalidate.

---

## 5. Option B — Morphology-Invariant Feature Subset (FALLBACK)

**Idea**: if retargeting doesn't fully remove the bias (or breaks in unexpected ways), strip morphology-dependent features from the AMP observation. Keep only features that are scalar/global and don't depend on joint frames.

### 5.1 Feature analysis

Current AMP obs (`amp_env.py` + `deepmimic_env.py:621-672`), with morphology sensitivity:

| Feature | Source | Morphology-invariant? |
|---------|--------|----------------------|
| `dof_pos` / `dof_vel` | engine | **Yes** (scalars per DOF) |
| `root_pos` / `root_rot` | engine | Yes (global) |
| `root_vel` / `root_ang_vel` | engine | **Yes** (global) |
| `root_height` | engine | **Yes** |
| `joint_rot` (policy) | `dof_to_rot(dof_pos)` via `Joint.axis` | **No** — joint axes rotate with `joint_X_p` |
| `joint_rot` (mocap) | pre-stored in motion file | **No** correspondence with policy's rotated axes |
| `body_pos` (policy) | engine FK with modified `joint_X_p` | **No** — depends on joint frames |
| `body_pos` (mocap) | `kin_char_model.FK` with fixed `_local_rotation` | **No** correspondence |
| `key_body_pos` | subset of body_pos | **No** |

### 5.2 Subset proposal

Drop: `joint_rot`, `body_pos`, `key_body_pos` (all the morphology-sensitive ones).
Keep: `dof_pos`, `dof_vel`, `root_pos_obs` (rel root pos), `root_rot_obs`, `root_vel_obs`, `root_ang_vel_obs`, `root_height_obs`.

### 5.3 Implementation

1. Subclass or modify `AMPEnv.compute_disc_obs()` / `_fetch_disc_demo_data()` to omit the listed features.
2. Keep the observation vector shapes consistent between policy and mocap streams.
3. Check `get_disc_obs_space()` — must reflect the reduced dim.
4. Retrain discriminator from scratch (weights are obs-shape-dependent).

### 5.4 Risks

- **Weaker prior**: with only DOF scalars and root motion, the discriminator has less to compare. May accept many more gaits as "natural" — we could end up back in the feet-dragging regime or at least with looser style constraints.
- **`dof_pos` is an under-determined signal**: two morphologies with different joint frames but the same DOF trajectory produce different end-effector motions. Mocap's "natural" DOF trajectory may not actually be natural on the modified morphology. (This is actually a reason to prefer Option A conceptually.)

### 5.5 Validation

Same Test 1 (bias), Test 3 (full run) as Option A. Test 2 is N/A — no FK comparison to run.

---

## 6. Sequencing

**Day 1 (tomorrow)**:
- [ ] §3 sanity check — confirm bias is material (1-2h)
- [ ] §4.2 step 1-3 — add `name_to_idx`, cache `_local_rotation`, implement `update_kin_char_local_rotations` (3-4h)
- [ ] §4.3 Test 2 — retargeting consistency unit test (1-2h)

**Day 2**:
- [ ] §4.2 step 4-5 — integrate into codesign loop (A1: θ-center-only retargeting first) (2-3h)
- [ ] §4.3 Test 1 — verify bias reduction (1h)
- [ ] Kick off a ~6h training run with retargeting enabled; compare CoT trajectory to the 12h baseline (start before leaving for the day, analyze day 3)

**Day 3**:
- [ ] Analyze day-2 run. Decision point:
  - CoT improving past plateau → success, continue to full-length 12h run for the paper
  - CoT still plateaus → move to A2 (per-perturbation retargeting) or B (feature subset)

**If A fails**:
- [ ] A2: per-perturbation retargeting (1-2 days)
- [ ] If A2 also fails: B (feature subset), ~1 day

---

## 7. Paper Implications (post-hoc, once implementation works)

If Option A succeeds, it adds a small but novel methodological contribution. Possible new section in the paper (§3.8 or inserted into §3.6):

> *"When using motion priors (AMP) in co-design, the discriminator's observation features are morphology-dependent. Without correction, morphology perturbations induce a systematic shift in `log D` unrelated to policy behavior, biasing the outer gradient. We address this by retargeting the motion-capture reference skeleton consistently with each morphology update: whenever θ is modified, we apply the same joint-frame delta to the kinematic model used for motion-prior forward kinematics. This restores the constrained Envelope Theorem's validity by ensuring the mocap reference tracks the current morphology."*

Cite Villegas et al. "Neural Kinematic Networks for Unsupervised Motion Retargetting" (CVPR 2018) or Aberman et al. "Skeleton-Aware Networks" (SIGGRAPH 2020) as prior work on morphology-aware retargeting.

The envelope-theorem section (§3.3) can then stay intact with one added sentence: *"Retargeting (§3.8) ensures the mocap-referenced component of the inner-loop objective tracks morphology, preserving the constrained optimum conditions under which the envelope approximation holds."*

---

## 8. References in the codebase (quick index)

| Purpose | File | Lines |
|---------|------|-------|
| AMP obs policy side | `MimicKit/mimickit/envs/amp_env.py` | 94-180 |
| AMP obs mocap side | `MimicKit/mimickit/envs/amp_env.py` | 63-86 |
| Mocap FK | `MimicKit/mimickit/anim/kin_char_model.py` | 176-208 |
| DOF→rot | `MimicKit/mimickit/anim/kin_char_model.py` | 48-61 (Joint), 146-158 (model) |
| Physics joint_X_p update | `codesign/codesign_g1_unified.py` | 426-452 |
| MJCF body quat modification | `codesign/g1_mjcf_modifier.py` | 91-121 |
| SPSA perturbation apply | `codesign/codesign_g1_unified.py` | 833-887 |
| SPSA gradient compute | `codesign/codesign_g1_unified.py` | 914-1124 |
| Design param list | `codesign/g1_mjcf_modifier.py` | 22-29 (`SYMMETRIC_PAIRS`) |
