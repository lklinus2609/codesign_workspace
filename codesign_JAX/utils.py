"""
Utility helpers for PGHC JAX: Adam optimizer, logging, checkpointing.
"""

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Adam optimizer for design parameters (numpy, gradient ascent)
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Adam optimizer for numpy arrays (gradient ascent on reward)."""

    def __init__(self, n_params, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params, grad):
        """Gradient ascent: params += lr * adapted_grad."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {"m": self.m.copy(), "v": self.v.copy(), "t": self.t,
                "lr": self.lr}

    def load_state_dict(self, d):
        self.m = d["m"].copy()
        self.v = d["v"].copy()
        self.t = d["t"]
        self.lr = d["lr"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, theta, adam_opt, policy_params, value_params,
                    obs_rms_state, outer_iter, extra=None):
    """Save full PGHC checkpoint."""
    ckpt = {
        "theta": np.asarray(theta),
        "adam_m": adam_opt.m.copy(),
        "adam_v": adam_opt.v.copy(),
        "adam_t": adam_opt.t,
        "policy_params": jax.tree.map(np.asarray, policy_params),
        "value_params": jax.tree.map(np.asarray, value_params),
        "obs_rms_state": obs_rms_state,
        "outer_iter": outer_iter,
    }
    if extra:
        ckpt.update(extra)
    np.savez(str(path), **{k: v for k, v in _flatten_dict(ckpt)})
    # Also save a readable JSON summary
    summary = {
        "outer_iter": int(outer_iter),
        "theta_deg": np.degrees(np.asarray(theta)).tolist(),
    }
    if extra:
        for k, v in extra.items():
            try:
                summary[k] = float(v)
            except (TypeError, ValueError):
                pass
    json_path = str(path).replace(".npz", ".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)


def _flatten_dict(d, prefix=""):
    """Flatten nested dict for np.savez."""
    items = []
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        elif isinstance(v, (np.ndarray, jnp.ndarray)):
            items.append((key, np.asarray(v)))
        elif isinstance(v, (int, float)):
            items.append((key, np.array(v)))
        else:
            items.append((key, np.array(v)))
    return items


# ---------------------------------------------------------------------------
# Running mean/std for observation normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for running mean/variance (numpy)."""

    def __init__(self, shape, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count *
                    batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, obs):
        """Normalize observation array. Works with numpy or jax arrays."""
        mean = self.mean.astype(np.float32)
        std = np.sqrt(self.var.astype(np.float32) + 1e-8)
        obs_norm = (np.asarray(obs) - mean) / std
        return np.clip(obs_norm, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(),
                "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"].copy()
        self.var = d["var"].copy()
        self.count = d["count"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class PGHCLogger:
    """Simple logger for PGHC training metrics."""

    def __init__(self, out_dir, use_wandb=False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.log_file = self.out_dir / "pghc_log.jsonl"
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print("[Logger] wandb not available, logging to file only")
                self.use_wandb = False

    def log(self, metrics, step=None):
        """Log metrics dict to file and optionally wandb."""
        entry = {"timestamp": time.time()}
        if step is not None:
            entry["step"] = step
        entry.update({k: float(v) if isinstance(v, (int, float, np.floating))
                      else v for k, v in metrics.items()})

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def init_wandb(self, project, name, config):
        """Initialize wandb run."""
        if self._wandb is not None:
            self._wandb.init(project=project, name=name, config=config)

    def finish(self):
        if self._wandb is not None:
            self._wandb.finish()
