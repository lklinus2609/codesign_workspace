# Orthogonal SPSA for Outer-Loop Morphology Gradient

## Problem with Current Coordinate-Wise FD

The current outer loop in `codesign_g1_unified.py` uses **coordinate-wise central finite differences**:

```
partition 0:        center (theta)
partition 2i+1:     theta with param i + eps
partition 2i+2:     theta with param i - eps
```

For N=6 symmetric params → 13 perturbations, each run with K paired seeds.

**Limitation**: each rollout only perturbs ONE parameter. The gradient estimate captures
isolated partial derivatives (dJ/dtheta_i) but **misses cross-parameter interactions** —
i.e., how changing one joint angle affects the optimal setting of another.

---

## SPSA (Simultaneous Perturbation Stochastic Approximation)

SPSA perturbs **all parameters at once** with a random direction vector Delta:

```
g_hat_i = (J(theta + eps*Delta) - J(theta - eps*Delta)) / (2*eps*Delta_i)
```

Each rollout carries information about all parameters. Averaging over M random directions
reconstructs the full gradient while capturing joint effects.

### Naive SPSA SNR Problem

For a single SPSA sample, Taylor expansion gives:

```
g_hat_i = dJ/dtheta_i  +  SUM_{j!=i} (dJ/dtheta_j * Delta_j/Delta_i)  +  rollout_noise
           ──────────      ────────────────────────────────────────────
             signal                     crosstalk
```

With Rademacher (+/-1) perturbations, crosstalk variance = SUM_{j!=i} (dJ/dtheta_j)^2.

If all partial derivatives have similar magnitude g:
- Per-sample SNR (ignoring rollout noise) = 1/sqrt(N-1) = 1/sqrt(5) ~ 0.45
- Below 1 for every single sample — unusable without heavy averaging
- Need M >= 9*(N-1) = 45 random directions just to reach SNR=3 from averaging

This makes naive random SPSA potentially **worse** than coordinate-wise FD for the same
compute budget.

---

## Solution: Orthogonal SPSA (QR-SPSA)

Instead of random Rademacher vectors, use **random orthogonal bases** as perturbation
directions.

### Algorithm

```python
from scipy.stats import special_ortho_group

S = 5   # number of orthogonal sets
N = 6   # number of parameters
K = 50  # paired seeds

directions = []
for _ in range(S):
    Q = special_ortho_group.rvs(N)   # random 6x6 orthogonal matrix
    directions.append(Q)              # each row = one perturbation direction

Delta = np.vstack(directions)  # shape (S*N, N) = (30, 6)
```

For each seed k:
1. Apply all 1 + 2*S*N perturbation morphologies via world-partitioning
2. Run one rollout (frozen policy, paired seed)
3. Collect scalar reward differences: `dJ[m] = (J_plus_m - J_minus_m) / (2*eps)`
4. Solve overdetermined system via least-squares:

```python
# Delta @ grad = dJ   (30 equations, 6 unknowns)
grad_k, _, _, _ = np.linalg.lstsq(Delta, dJ, rcond=None)
```

Final gradient = mean of grad_k over K seeds.

### Why Orthogonality Fixes the SNR Problem

Within each orthogonal set of N directions, the crosstalk terms **cancel exactly** when
averaged — same SNR as coordinate-wise FD. But the directions are randomly rotated, so
they probe the parameter space from different angles and **capture cross-parameter coupling**.

Multiple orthogonal sets (S > 1) provide additional overdetermined averaging with the
crosstalk-cancellation guarantee preserved within each set.

---

## Compute Budget (4x A100/H100, 16k envs per GPU)

### World Partitioning

```
n_pert = 1 + 2*S*N     (1 center + 2 per direction)
wpp = 16000 / n_pert    (worlds per perturbation per GPU)
```

| S (orth sets) | Directions | n_pert | wpp/GPU | wpp effective (x4 GPU) |
|---------------|-----------|--------|---------|------------------------|
| 1             | 6         | 13     | 1230    | 4920                   |
| 2             | 12        | 25     | 640     | 2560                   |
| 3             | 18        | 37     | 432     | 1728                   |
| 4             | 24        | 49     | 326     | 1304                   |
| 5             | 30        | 61     | 262     | 1048                   |
| 6             | 36        | 73     | 219     | 876                    |

### Recommended Configuration: S=5, K=50

- **30 orthogonal directions** — 5 complete sets of 6, diverse coverage of 6D space
- **wpp=262 per GPU** — sufficient for stable reward estimates (~1048 effective across 4 GPUs)
- **K=50 paired seeds** — strong variance reduction from seed averaging
- **1500 total gradient samples** for 6 unknowns — massively overdetermined
- **Wall-clock**: 50 sequential rollouts per outer iteration (~4-8 min on 4x A100)

---

## Gradient Reconstruction

```python
# Per seed: solve overdetermined linear system
grad_per_seed = []  # list of (6,) arrays

for k in range(K):
    dJ = np.array([
        (reward_plus[m, k] - reward_minus[m, k]) / (2 * eps)
        for m in range(S * N)
    ])  # shape (30,)

    grad_k, _, _, _ = np.linalg.lstsq(Delta, dJ, rcond=None)
    grad_per_seed.append(grad_k)

grad_samples = np.stack(grad_per_seed)  # (50, 6)
grad = grad_samples.mean(axis=0)        # (6,)
stderr = grad_samples.std(axis=0, ddof=1) / np.sqrt(K)  # (6,)
snr = np.abs(grad) / stderr             # (6,)
```

---

## Safe Morphology Update with SNR Gating

Only update parameters where the gradient has sufficient statistical confidence:

```python
SNR_THRESHOLD = 2.0  # ~95% confidence the sign is correct

mask = snr > SNR_THRESHOLD
theta_new = theta + lr * grad * mask.astype(float)
theta_new = np.clip(theta_new, -np.radians(30), np.radians(30))
```

This prevents taking morphology steps on noisy gradient components. With K=50 seeds
you have real statistical power to distinguish signal from noise.

### Logging

Per outer iteration, log:
- Per-parameter: gradient, stderr, SNR, whether update was applied
- Overall: center reward (CoT + vel tracking), number of params updated

---

## Implementation Changes to `codesign_g1_unified.py`

### What Changes

1. **`apply_partitioned_morphologies`**: Replace axis-aligned perturbations with
   orthogonal direction perturbations. Each partition applies `theta + eps * Delta[m]`
   instead of `theta + eps * e_i`.

2. **`compute_fd_gradient_parallel`**: Replace per-parameter finite difference formula
   with least-squares solve over all directions.

3. **New function**: `generate_orthogonal_directions(N, S)` — returns (S*N, N) matrix
   of perturbation directions from S random orthogonal bases.

4. **Argument changes**:
   - Replace `--fd-seeds` with `--spsa-seeds` (K, default 50)
   - Add `--spsa-sets` (S, default 5)
   - Keep `--fd-epsilon` (eps, default 0.05)

### What Stays the Same

- World-partitioning architecture (just different morphologies per partition)
- Paired-seed structure for variance reduction
- Inner loop (PPO training) unchanged
- Morphology parameterization (symmetric joint angles, theta -> quaternion)
- SGD outer update + clipping to +/-30 deg
- Multi-GPU all_reduce averaging

---

## Variance Comparison Summary

| Method                  | Crosstalk | Cross-param coupling | Rollouts/seed | SNR scaling         |
|-------------------------|-----------|---------------------|---------------|---------------------|
| Coordinate-wise FD      | None      | No                  | 2N+1 = 13    | sqrt(K) / sigma     |
| Naive SPSA (Rademacher) | High      | Yes                 | 2M            | sqrt(M*K) / sigma'  |
| Orthogonal SPSA (QR)    | Cancels   | Yes                 | 2*S*N         | sqrt(S*K) / sigma   |

Orthogonal SPSA achieves coordinate-wise-FD-level SNR while capturing the cross-parameter
interactions that coordinate-wise FD misses.
