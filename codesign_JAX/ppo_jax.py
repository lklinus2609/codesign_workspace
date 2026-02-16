"""
PPO Inner Loop in Pure JAX.

MLP policy/value networks with diagonal Gaussian actions.
Vectorized rollouts via jax.vmap, training via jax.lax.scan.
Convergence detection via reward plateau.

Uses Flax for neural network parameterization.
"""

import functools
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


# ---------------------------------------------------------------------------
# Neural network definitions
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP with ELU activations."""
    features: Sequence[int]
    final_activation: bool = False

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=nn.initializers.lecun_normal())(x)
            if i < len(self.features) - 1 or self.final_activation:
                x = nn.elu(x)
        return x


class PolicyNetwork(nn.Module):
    """Gaussian policy: MLP mean + learned log_std."""
    hidden_dims: Sequence[int] = (256, 128, 128)
    act_dim: int = 29

    @nn.compact
    def __call__(self, obs):
        mean = MLP((*self.hidden_dims, self.act_dim))(obs)
        # log_std as a learnable parameter (not input-dependent)
        log_std = self.param("log_std", nn.initializers.constant(0.0),
                             (self.act_dim,))
        return mean, log_std


class ValueNetwork(nn.Module):
    """Value function: MLP -> scalar."""
    hidden_dims: Sequence[int] = (256, 128, 128)

    @nn.compact
    def __call__(self, obs):
        value = MLP((*self.hidden_dims, 1))(obs)
        return value.squeeze(-1)


# ---------------------------------------------------------------------------
# PPO data structures
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Single transition for PPO rollout."""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


class PPOConfig(NamedTuple):
    """PPO hyperparameters (IsaacLab G1 flat defaults)."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.008
    value_coeff: float = 1.0
    max_grad_norm: float = 1.0
    lr: float = 1e-3
    n_epochs: int = 5
    num_minibatches: int = 4
    horizon: int = 24
    num_envs: int = 4096
    desired_kl: float = 0.01


# ---------------------------------------------------------------------------
# Core PPO functions
# ---------------------------------------------------------------------------

def init_networks(rng, obs_dim, act_dim, hidden_dims=(256, 256)):
    """Initialize policy and value networks.

    Returns:
        policy_params, value_params, policy_apply, value_apply
    """
    policy = PolicyNetwork(hidden_dims=hidden_dims, act_dim=act_dim)
    value = ValueNetwork(hidden_dims=hidden_dims)

    rng_p, rng_v = jax.random.split(rng)
    dummy_obs = jnp.zeros(obs_dim)
    policy_params = policy.init(rng_p, dummy_obs)
    value_params = value.init(rng_v, dummy_obs)

    return policy_params, value_params, policy.apply, value.apply


def sample_action(policy_apply, policy_params, obs, rng):
    """Sample action from policy distribution.

    Args:
        policy_apply: policy network apply function
        policy_params: policy parameters
        obs: (obs_dim,) observation
        rng: PRNGKey

    Returns:
        action: (act_dim,) clipped action
        log_prob: scalar log probability
        value: not returned here (separate call)
    """
    mean, log_std = policy_apply(policy_params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape)
    action = mean + std * noise
    action_clipped = jnp.clip(action, -1.0, 1.0)

    # Log probability under the Gaussian (before clipping, for PPO ratio)
    log_prob = -0.5 * jnp.sum(
        jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi)
    )
    return action_clipped, log_prob


def deterministic_action(policy_apply, policy_params, obs):
    """Deterministic action (mean of policy distribution)."""
    mean, _ = policy_apply(policy_params, obs)
    return jnp.clip(mean, -1.0, 1.0)


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (H, N) rewards
        values: (H, N) value predictions
        dones: (H, N) done flags
        last_value: (N,) bootstrap value
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (H, N)
        returns: (H, N)
    """
    H, N = rewards.shape

    def _gae_step(carry, t):
        """Reverse scan step for GAE computation."""
        last_gae, next_val = carry
        done = dones[t]
        reward = rewards[t]
        value = values[t]
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * last_gae
        return (gae, value), gae

    # Scan in reverse
    init_carry = (jnp.zeros(N), last_value)
    _, advantages = jax.lax.scan(
        _gae_step, init_carry,
        jnp.arange(H - 1, -1, -1),  # reverse indices
    )
    # Reverse back to forward order
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_loss(policy_params, value_params, policy_apply, value_apply,
             batch, clip_ratio, entropy_coeff, value_coeff):
    """Compute PPO loss for a minibatch.

    Args:
        policy_params, value_params: network parameters
        policy_apply, value_apply: network apply functions
        batch: dict with obs, actions, old_log_probs, advantages, returns
        clip_ratio: PPO clipping parameter
        entropy_coeff: entropy bonus coefficient
        value_coeff: value loss coefficient

    Returns:
        total_loss, info_dict
    """
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    # Policy forward
    def _policy_single(obs_i, act_i):
        mean, log_std = policy_apply(policy_params, obs_i)
        std = jnp.exp(log_std)
        log_prob = -0.5 * jnp.sum(
            jnp.square((act_i - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        entropy = 0.5 * jnp.sum(1 + 2 * log_std + jnp.log(2 * jnp.pi))
        return log_prob, entropy

    log_probs, entropies = jax.vmap(_policy_single)(obs, actions)

    # Value forward
    values = jax.vmap(value_apply, in_axes=(None, 0))(value_params, obs)

    # PPO clipped objective
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages,
                                         clipped_ratio * advantages))

    # Value loss
    value_loss = jnp.mean(jnp.square(values - returns))

    # Entropy bonus
    entropy = jnp.mean(entropies)

    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    info = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jnp.mean(old_log_probs - log_probs),
    }
    return total_loss, info


# ---------------------------------------------------------------------------
# PPO training step
# ---------------------------------------------------------------------------

def make_ppo_train_fns(policy_apply, value_apply, ppo_cfg):
    """Create JIT-compiled PPO training functions.

    Args:
        policy_apply: policy network apply function
        value_apply: value network apply function
        ppo_cfg: PPOConfig

    Returns:
        dict with 'collect_rollout_step', 'update' functions
    """
    gamma = ppo_cfg.gamma
    gae_lambda = ppo_cfg.gae_lambda
    clip_ratio = ppo_cfg.clip_ratio
    entropy_coeff = ppo_cfg.entropy_coeff
    value_coeff = ppo_cfg.value_coeff
    max_grad_norm = ppo_cfg.max_grad_norm
    n_epochs = ppo_cfg.n_epochs
    num_minibatches = ppo_cfg.num_minibatches

    @jax.jit
    def update(policy_params, value_params, opt_state, transitions, last_value,
               rng):
        """Run PPO update on collected transitions.

        Args:
            policy_params, value_params: current network parameters
            opt_state: optimizer state
            transitions: Transition with (H, N, ...) arrays
            last_value: (N,) bootstrap value
            rng: PRNGKey

        Returns:
            new_policy_params, new_value_params, new_opt_state, info
        """
        # Compute GAE
        advantages, returns = compute_gae(
            transitions.reward, transitions.value, transitions.done,
            last_value, gamma, gae_lambda,
        )

        # Normalize advantages
        adv_mean = jnp.mean(advantages)
        adv_std = jnp.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        H, N = transitions.reward.shape
        total_samples = H * N
        minibatch_size = total_samples // num_minibatches

        # Flatten time and env dimensions
        flat_obs = transitions.obs.reshape(total_samples, -1)
        flat_actions = transitions.action.reshape(total_samples, -1)
        flat_log_probs = transitions.log_prob.reshape(total_samples)
        flat_advantages = advantages.reshape(total_samples)
        flat_returns = returns.reshape(total_samples)

        def _epoch(carry, rng_epoch):
            policy_params, value_params, opt_state = carry

            # Shuffle
            perm = jax.random.permutation(rng_epoch, total_samples)

            def _minibatch(carry, mb_idx):
                policy_params, value_params, opt_state = carry
                start = mb_idx * minibatch_size
                idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))

                batch = {
                    "obs": flat_obs[idx],
                    "actions": flat_actions[idx],
                    "log_probs": flat_log_probs[idx],
                    "advantages": flat_advantages[idx],
                    "returns": flat_returns[idx],
                }

                grad_fn = jax.grad(ppo_loss, argnums=(0, 1), has_aux=True)
                (p_grad, v_grad), info = grad_fn(
                    policy_params, value_params, policy_apply, value_apply,
                    batch, clip_ratio, entropy_coeff, value_coeff,
                )

                # Clip gradients
                p_grad = optax.clip_by_global_norm(max_grad_norm).update(
                    p_grad, optax.EmptyState())[0]
                v_grad = optax.clip_by_global_norm(max_grad_norm).update(
                    v_grad, optax.EmptyState())[0]

                # Combine gradients
                grads = (p_grad, v_grad)
                params = (policy_params, value_params)
                updates, new_opt_state = optimizer.update(grads, opt_state,
                                                          params)
                new_params = optax.apply_updates(params, updates)
                new_policy_params, new_value_params = new_params

                return (new_policy_params, new_value_params, new_opt_state), info

            carry, infos = jax.lax.scan(
                _minibatch, carry, jnp.arange(num_minibatches)
            )
            return carry, infos

        # Create optimizer
        rng_epochs = jax.random.split(rng, n_epochs)
        carry = (policy_params, value_params, opt_state)
        carry, epoch_infos = jax.lax.scan(_epoch, carry, rng_epochs)
        new_policy_params, new_value_params, new_opt_state = carry

        # Aggregate info from last epoch, last minibatch
        final_info = jax.tree.map(lambda x: x[-1, -1], epoch_infos)

        return new_policy_params, new_value_params, new_opt_state, final_info

    # Create optimizer (shared for policy + value)
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(ppo_cfg.lr),
    )

    def init_optimizer(policy_params, value_params):
        """Initialize optimizer state."""
        params = (policy_params, value_params)
        return optimizer.init(params)

    return {
        "update": update,
        "optimizer": optimizer,
        "init_optimizer": init_optimizer,
    }


# ---------------------------------------------------------------------------
# Rollout collection (called from training loop, not JIT-compiled standalone)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2, 3))
def collect_rollout(env_states, policy_params, policy_apply, value_apply,
                    value_params, rng, horizon):
    """Collect a rollout of H steps across N environments.

    Note: This function steps the environment inside a jax.lax.scan.
    env_fns must be passed as static or captured in a closure.

    This version takes env_states and returns transitions + new states.
    The actual env stepping is done outside via the returned actions.
    """
    # This is a helper that just batches the action sampling
    def _sample_batch(policy_params, obs_batch, rng):
        rngs = jax.random.split(rng, obs_batch.shape[0])
        actions, log_probs = jax.vmap(
            sample_action, in_axes=(None, None, 0, 0)
        )(policy_apply, policy_params, obs_batch, rngs)
        values = jax.vmap(value_apply, in_axes=(None, 0))(
            value_params, obs_batch)
        return actions, log_probs, values

    return _sample_batch


def collect_actions_deterministic(policy_apply, policy_params, obs_batch):
    """Collect deterministic actions for BPTT evaluation.

    Args:
        policy_apply: policy network apply function
        policy_params: policy parameters
        obs_batch: (N, obs_dim) observations

    Returns:
        (N, act_dim) deterministic actions
    """
    return jax.vmap(deterministic_action, in_axes=(None, None, 0))(
        policy_apply, policy_params, obs_batch,
    )


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

class StabilityGate:
    """Convergence detection: reward plateau over a window."""

    def __init__(self, rel_threshold=0.02, min_iters=500,
                 stable_iters_required=50, window=5):
        self.rel_threshold = rel_threshold
        self.min_iters = min_iters
        self.stable_iters_required = stable_iters_required
        self.window = window
        self.reward_history = []
        self.total_iters = 0
        self.stable_count = 0

    def reset(self):
        self.reward_history = []
        self.total_iters = 0
        self.stable_count = 0

    def update(self, mean_reward):
        self.reward_history.append(float(mean_reward))
        if len(self.reward_history) > self.window:
            self.reward_history = self.reward_history[-self.window:]

    def tick(self, n=1):
        self.total_iters += n
        if self._is_plateau():
            self.stable_count += n
        else:
            self.stable_count = 0

    def _is_plateau(self):
        if len(self.reward_history) < 2:
            return False
        import numpy as np
        vals = np.array(self.reward_history)
        mean_val = np.mean(vals)
        if abs(mean_val) < 1e-6:
            return True
        spread = np.max(vals) - np.min(vals)
        return (spread / abs(mean_val)) < self.rel_threshold

    def is_converged(self):
        return (self.total_iters >= self.min_iters and
                self.stable_count >= self.stable_iters_required)
