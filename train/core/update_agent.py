import train.mytypes as train_types
from agents import BaseAgent

from typing import Any, Literal, Optional, Tuple, Union
from functools import partial

import jax
import chex
import jax.numpy as jnp
from flax import nnx

@chex.dataclass
class UpdateState:
    agent: BaseAgent
    optimizer: nnx.Optimizer
    metrics: nnx.MultiMetric
    key: chex.PRNGKey

@partial(nnx.jit, static_argnames=('num_minibatches', 'num_ppo_epoch', 'only_use_player0_experience', 'mag_divergence_type'))
def update_agent(
    agent: BaseAgent,
    mag_agent: BaseAgent,
    optimizer: nnx.Optimizer,
    dataset: train_types.Dataset,
    metrics: nnx.MultiMetric,
    key: chex.PRNGKey,
    ent_coef: float,
    clip_eps: float,
    num_minibatches: int,
    num_ppo_epoch: int,
    only_use_player0_experience: bool,
    mag_coef: float = 0.0,
    mag_divergence_type: Literal["kl", "l2"] = "kl",
) -> Tuple[BaseAgent, nnx.Optimizer, nnx.MultiMetric]:
    batch_size = dataset.advantage.shape[0]

    assert batch_size % num_minibatches == 0, f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})"

    # mask the dataset if specified
    if only_use_player0_experience:
        # only use data where it is valid and is act by player 0
        dataset.valid_mask = jnp.logical_and(dataset.valid_mask, dataset.current_player == jnp.int32(0))

    def masked_mean(x: chex.Array, mask: chex.Array) -> chex.Numeric:
        """Compute mean only over valid (masked) samples"""
        masked_x = x * mask
        return jnp.sum(masked_x) / jnp.maximum(jnp.sum(mask), 1.0)
    
    def masked_var(x: chex.Array, mask: chex.Array) -> chex.Numeric:
        """Compute variance only over valid (masked) samples"""
        masked_mean_val = masked_mean(x, mask)
        masked_variance = jnp.sum(mask * jnp.square(x - masked_mean_val)) / jnp.maximum(jnp.sum(mask), 1.0)
        return masked_variance
    
    def masked_std(x: chex.Array, mask: chex.Array) -> chex.Numeric:
        """Compute standard deviation only over valid (masked) samples"""
        return jnp.sqrt(masked_var(x, mask))
    

    def calculate_n_log_loss(
        agent: BaseAgent, dataset: train_types.Dataset, metrics: nnx.MultiMetric
    ) -> chex.Numeric:
        """calculate loss and log to metrics"""
        dists = agent.get_action_distribution(dataset.observation, dataset.action_mask)
        
        """actor loss"""
        log_prob = dists.log_prob(dataset.action)
        
        # normalize advantage using valid mask
        advantage_mean = masked_mean(dataset.advantage, dataset.valid_mask)
        advantage_std = masked_std(dataset.advantage, dataset.valid_mask)
        dataset.advantage = (dataset.advantage - advantage_mean) / (advantage_std + 1e-8)

        # ppo loss
        log_ratio = log_prob - dataset.log_prob
        ratio = jnp.exp(log_ratio)
        ppo_loss1 = ratio * dataset.advantage
        ppo_loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * dataset.advantage
        ppo_loss = -masked_mean(jnp.minimum(ppo_loss1, ppo_loss2), dataset.valid_mask)

        # entropy loss
        entropy_loss = -masked_mean(dists.entropy(), dataset.valid_mask)

        # magnet loss
        # If mag_agent is provided: KL/L2 toward that fixed agent (Nash PG style).
        # If mag_agent is None: backward KL toward π_old (MMD / paper Section J style).
        mag_loss, mag_kl = 0, 0
        if mag_agent is not None:
            mag_dists = mag_agent.get_action_distribution(dataset.observation, dataset.action_mask)
            if mag_divergence_type == "kl":
                mag_kl = masked_mean(dists.kl_divergence(mag_dists), dataset.valid_mask)
            elif mag_divergence_type == "l2":
                probs = dists.probs
                mag_probs = mag_dists.probs
                mag_kl = 0.5 * masked_mean(jnp.sum(jnp.square(probs - mag_probs), axis=-1), dataset.valid_mask)
            mag_loss = mag_kl
        else:
            # Backward (reverse) KL: KL(π_new || π_old) ≈ ratio * log_ratio - (ratio - 1)
            # See http://joschu.net/blog/kl-approx.html
            backward_kl = ratio * log_ratio - (ratio - 1)
            mag_kl = masked_mean(backward_kl, dataset.valid_mask)
            mag_loss = mag_kl

        # total actor loss
        actor_loss = ppo_loss + ent_coef * entropy_loss + mag_coef * mag_loss

        """critic loss"""
        values = agent.get_value(dataset.observation)
        values_clipped = dataset.value + jnp.clip(values - dataset.value, -clip_eps, clip_eps)
        critic_loss1 = jnp.square(values - dataset.target_value)
        critic_loss2 = jnp.square(values_clipped - dataset.target_value)
        critic_loss = 0.5 * masked_mean(jnp.maximum(critic_loss1, critic_loss2), dataset.valid_mask)

        """logging"""
        total_loss = actor_loss + critic_loss
        approx_kl = masked_mean((ratio - 1) - log_ratio, dataset.valid_mask)
        clip_frac = masked_mean((jnp.abs(ratio - 1.0) > clip_eps).astype('float32'), dataset.valid_mask)
        
        # explained variance calculation with masking
        target_var = masked_var(dataset.target_value, dataset.valid_mask)
        residual_var = masked_var(dataset.target_value - values, dataset.valid_mask)
        explained_var = jnp.maximum(1 - residual_var / (target_var + 1e-8), jnp.float32(0))

        metrics.update(
            actor_loss = actor_loss,
            ppo_loss = ppo_loss,
            critic_loss=critic_loss,
            entropy = -entropy_loss,
            mag_kl = mag_kl,
            approx_kl = approx_kl,
            clip_frac = clip_frac,
            explained_var = explained_var
        )

        return total_loss


    def update_batch(carry: UpdateState, batch: train_types.Dataset):
        """Update the agent for a single batch"""
        # compute the gradient
        grad = nnx.grad(calculate_n_log_loss)(carry.agent, batch, carry.metrics)

        # update agent, optimizer state (inplace update)
        carry.optimizer.update(grad)

        return carry, 0


    def update_epoch(carry: UpdateState, _: Any):
        """Update the agent for a single epoch"""
        carry.key, shuffle_key1 = jax.random.split(carry.key, 2)

        # Shuffle data and create minibatches
        permutation1 = jax.random.permutation(shuffle_key1, batch_size)
        def process_batch1(x: chex.Array):
            # shuffle
            x = jnp.take(x, permutation1, axis=0)
            # create mini-batches
            x = jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))
            return x
        batched_dataset = jax.tree.map(process_batch1, dataset) # (num_minibatches, batch_size, ...)

        # update batches
        carry, _ = nnx.scan(update_batch)(carry, batched_dataset)

        return carry, 0


    # create update state for carrying
    carry = UpdateState(
        agent=agent,
        optimizer=optimizer,
        metrics=metrics,
        key=key
    )

    # perform ppo update for given epoch
    carry, _ = nnx.scan(update_epoch, length=num_ppo_epoch)(carry, None)
    return carry.agent, carry.optimizer, carry.metrics


@partial(nnx.jit, static_argnames=('num_minibatches', 'num_epoch'))
def update_distill_agent(
    distill_agent: BaseAgent,
    optimizer: nnx.Optimizer,
    dataset: train_types.Dataset,
    metrics: nnx.MultiMetric,
    key: chex.PRNGKey,
    num_minibatches: int,
    num_epoch: int,
) -> Tuple[BaseAgent, nnx.Optimizer, nnx.MultiMetric]:
    """Behavioral cloning: train distill_agent to mimic opponent (player 1) actions."""
    batch_size = dataset.action.shape[0]

    def bc_loss(agent: BaseAgent, dataset: train_types.Dataset, metrics: nnx.MultiMetric) -> chex.Numeric:
        opp_mask = (dataset.current_player == jnp.int32(1)).astype(jnp.float32)
        dist = agent.get_action_distribution(dataset.observation, dataset.action_mask)
        log_prob = dist.log_prob(dataset.action)
        num_valid = jnp.maximum(jnp.sum(opp_mask), 1.0)
        loss = -jnp.sum(log_prob * opp_mask) / num_valid
        metrics.update(bc_loss=loss)
        return loss

    def update_batch(carry: UpdateState, batch: train_types.Dataset):
        grad = nnx.grad(bc_loss)(carry.agent, batch, carry.metrics)
        carry.optimizer.update(grad)
        return carry, 0

    def update_epoch(carry: UpdateState, _: Any):
        carry.key, shuffle_key = jax.random.split(carry.key)
        perm = jax.random.permutation(shuffle_key, batch_size)
        def process(x: chex.Array):
            x = jnp.take(x, perm, axis=0)
            return jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))
        batched = jax.tree.map(process, dataset)
        carry, _ = nnx.scan(update_batch)(carry, batched)
        return carry, 0

    carry = UpdateState(agent=distill_agent, optimizer=optimizer, metrics=metrics, key=key)
    carry, _ = nnx.scan(update_epoch, length=num_epoch)(carry, None)
    return carry.agent, carry.optimizer, carry.metrics