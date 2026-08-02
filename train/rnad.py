"""
R-NaD (Regularized Nash Dynamics) training script.
https://arxiv.org/abs/2206.15378

Double loop structure:
  Outer loop: num_outer_update iterations — anchor policy is fixed per outer step
  Inner loop: num_inner_update iterations per outer step
    Each inner step: collect one batch of trajectories, then run num_epoch
    gradient updates with IS correction for the off-policy epochs.
"""

import os
from pathlib import Path
from typing import Any, Tuple
import logging

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
logging.getLogger('absl').setLevel(logging.WARNING)

from functools import partial
from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax import nnx
import chex
import optax
import hydra
from omegaconf import DictConfig

from envs import create_env
import envs.mytypes as env_types
from agents import create_agent, BaseAgent
from train.core.rnad_losses import (
    legal_log_policy, _player_others, _policy_ratio,
    v_trace, get_loss_v, get_loss_nerd,
)
from train.loggers import create_logger, BaseLogger


# ---------------------------------------------------------------------------
# Trajectory data structure (extends Transition with behavior policy probs)
# ---------------------------------------------------------------------------

@chex.dataclass
class RNaDTransition:
    is_new_eps: chex.Array      # (num_envs, num_steps) bool
    action: chex.Array          # (num_envs, num_steps)
    value: chex.Array           # (num_envs, num_steps)
    reward: chex.Array          # (num_envs, num_steps, num_agents)
    log_prob: chex.Array        # (num_envs, num_steps)
    observation: Any            # (num_envs, num_steps, obs_shape)
    action_mask: chex.Array     # (num_envs, num_steps, num_actions)
    current_player: chex.Array  # (num_envs, num_steps)
    policy: chex.Array          # (num_envs, num_steps, num_actions) — behavior policy


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@partial(nnx.jit, static_argnames=('env', 'num_envs', 'num_steps'))
def collect_rnad_trajectories(
    env: env_types.BaseEnv,
    agent: BaseAgent,
    env_state: env_types.EnvState,
    last_timestep: env_types.TimeStep,
    key: chex.PRNGKey,
    num_envs: int,
    num_steps: int,
) -> Tuple[env_types.EnvState, env_types.TimeStep, RNaDTransition]:
    """Collect trajectories and store the full behavior policy probability vector."""

    def collect_one_step(carry, _):
        last_ts, env_st, key = carry
        key, act_key = jax.random.split(key)

        # Add batch dim for agent inference
        ts_batched = jax.tree.map(lambda x: jnp.expand_dims(x, 0), last_ts)

        dist = agent.get_action_distribution(ts_batched.observation, ts_batched.action_mask)
        action, log_prob = dist.sample_and_log_prob(seed=act_key)
        policy = dist.probs  # (1, A) — full probability vector
        value = agent.get_value(ts_batched.observation)  # (1,)

        # Remove batch dim
        action = jnp.squeeze(action, 0)
        log_prob = jnp.squeeze(log_prob, 0)
        policy = jnp.squeeze(policy, 0)  # (A,)
        value = jnp.squeeze(value, 0)

        # Step env
        env_st, new_ts = env.step(env_st, action)

        transition = RNaDTransition(
            is_new_eps=last_ts.done,
            action=action,
            value=value,
            reward=new_ts.reward,
            log_prob=log_prob,
            observation=last_ts.observation,
            action_mask=last_ts.action_mask,
            current_player=last_ts.current_player,
            policy=policy,
        )
        return (new_ts, env_st, key), transition

    # Scan over num_steps per env, vmap over num_envs
    single_env_rollout = nnx.scan(collect_one_step, length=num_steps)
    batch_rollout = nnx.vmap(single_env_rollout, in_axes=(0, None), out_axes=0)

    keys = jax.random.split(key, num_envs)
    (last_timestep, env_state, _), trajectories = batch_rollout(
        (last_timestep, env_state, keys), None
    )
    return env_state, last_timestep, trajectories


# ---------------------------------------------------------------------------
# RNAD loss (called inside nnx.grad)
# ---------------------------------------------------------------------------

def compute_rnad_loss(
    agent: BaseAgent,
    anchor_agent: BaseAgent,  # captured in closure — not differentiated
    trajectory: RNaDTransition,
    metrics: nnx.MultiMetric,
    eta: float,
    c_vtrace: float,
    nerd_clip: float,
    nerd_beta: float,
    gamma: float,
) -> chex.Numeric:
    """Full RNAD loss for one batch.

    Args:
        agent: current policy (differentiated)
        anchor_agent: fixed anchor for entropy regularization (not differentiated)
        trajectory: shape (num_envs, num_steps, ...)
        metrics: updated in-place
        eta, c_vtrace, nerd_clip, nerd_beta: RNAD hyperparameters

    Returns:
        scalar loss
    """
    B, T = trajectory.current_player.shape
    num_actions = trajectory.action_mask.shape[-1]

    # Flatten (B, T, ...) -> (B*T, ...) for batched agent inference
    flat_obs = jax.tree.map(lambda x: x.reshape(B * T, *x.shape[2:]), trajectory.observation)
    flat_mask = trajectory.action_mask.reshape(B * T, num_actions)  # (B*T, A)

    # Current policy
    dist = agent.get_action_distribution(flat_obs, flat_mask)
    pi = dist.probs.reshape(B, T, num_actions)             # (B, T, A)
    # Safe logits: replace -inf (masked illegal) with 0 for NERD computation
    safe_logits = jnp.where(flat_mask, dist.logits, 0.0).reshape(B, T, num_actions)
    v = agent.get_value(flat_obs).reshape(B, T, 1)         # (B, T, 1)

    # log(pi_current / pi_anchor) for all actions (0 for illegal)
    log_pi = legal_log_policy(dist.logits.reshape(B, T, num_actions), flat_mask.reshape(B, T, num_actions))
    anchor_dist = anchor_agent.get_action_distribution(flat_obs, flat_mask)
    log_pi_anchor = legal_log_policy(
        anchor_dist.logits.reshape(B, T, num_actions), flat_mask.reshape(B, T, num_actions)
    )
    log_policy_reg = log_pi - log_pi_anchor  # (B, T, A)

    # Transpose to (T, B, ...) for v-trace scan
    def tb(x):
        return jnp.swapaxes(x, 0, 1)

    player_id = tb(trajectory.current_player)   # (T, B)
    is_new_eps = tb(trajectory.is_new_eps)       # (T, B)
    actions_oh = tb(jax.nn.one_hot(trajectory.action, num_actions))  # (T, B, A)
    pi_TB = tb(pi)                               # (T, B, A)
    log_policy_reg_TB = tb(log_policy_reg)       # (T, B, A)
    v_TB = tb(v)                                 # (T, B, 1)
    behavior_policy_TB = tb(trajectory.policy)   # (T, B, A)
    safe_logits_TB = tb(safe_logits)             # (T, B, A)
    legal_TB = tb(flat_mask.reshape(B, T, num_actions))  # (T, B, A)

    # V-trace for each player
    v_target_list, has_played_list, q_vr_list = [], [], []
    for player in range(2):
        reward_TB = tb(trajectory.reward[..., player])  # (T, B)
        player_others_TB = _player_others(player_id, player)  # (T, B, 1)

        vt, hp, q = v_trace(
            v=v_TB,
            is_new_eps=is_new_eps,
            player_id=player_id,
            acting_policy=behavior_policy_TB,
            merged_policy=pi_TB,
            merged_log_policy=log_policy_reg_TB,
            player_others=player_others_TB,
            actions_oh=actions_oh,
            reward=reward_TB,
            player=player,
            eta=eta,
            lambda_=1.0,
            c=c_vtrace,
            rho=jnp.inf,
            gamma=gamma,
        )
        v_target_list.append(vt)
        has_played_list.append(hp)
        q_vr_list.append(q)

    loss_v = get_loss_v([v_TB] * 2, v_target_list, has_played_list)

    # IS correction: pi_current(a) / mu(a) for each step
    mu_a = jnp.sum(behavior_policy_TB * actions_oh, axis=-1)       # (T, B)
    pi_a = jnp.sum(pi_TB * actions_oh, axis=-1)                    # (T, B)
    is_c = jnp.expand_dims(pi_a / (mu_a + 1e-8), axis=-1)          # (T, B, 1)

    loss_nerd = get_loss_nerd(
        logit_list=[safe_logits_TB] * 2,
        policy_list=[pi_TB] * 2,
        q_vr_list=q_vr_list,
        player_ids=player_id,
        legal_actions=legal_TB,
        importance_sampling_correction=[is_c] * 2,
        clip=nerd_clip,
        threshold=nerd_beta,
    )

    total_loss = loss_v + loss_nerd
    metrics.update(loss_v=loss_v, loss_nerd=loss_nerd, total_loss=total_loss)
    return total_loss


# ---------------------------------------------------------------------------
# Learner state and training steps
# ---------------------------------------------------------------------------

@chex.dataclass
class LearnerState:
    key: chex.PRNGKey
    env_state: env_types.EnvState
    last_timestep: env_types.TimeStep
    agent: BaseAgent
    optimizer: nnx.Optimizer
    train_metrics: nnx.MultiMetric
    rollout_metrics: nnx.MultiMetric
    anchor_agent: BaseAgent


@chex.dataclass
class UpdateState:
    agent: BaseAgent
    optimizer: nnx.Optimizer
    metrics: nnx.MultiMetric
    key: chex.PRNGKey


@partial(nnx.jit, static_argnames=('env', 'config'))
def single_training_step(
    learner_state: LearnerState,
    _: Any,
    env: env_types.BaseEnv,
    config: DictConfig,
) -> Tuple[LearnerState, Any]:
    """One inner training step: collect one batch, run num_epoch gradient updates."""

    # Collect trajectories
    learner_state.key, collect_key = jax.random.split(learner_state.key)
    learner_state.env_state, learner_state.last_timestep, trajectory = collect_rnad_trajectories(
        env=env,
        agent=learner_state.agent,
        env_state=learner_state.env_state,
        last_timestep=learner_state.last_timestep,
        key=collect_key,
        num_envs=config.algorithm.num_envs,
        num_steps=config.algorithm.num_steps,
    )

    # Log rollout metrics
    B, T = trajectory.current_player.shape
    learner_state.rollout_metrics.update(
        inverse_eps_len=trajectory.is_new_eps.reshape(B * T),
        reward=trajectory.reward.reshape(B * T, -1)[:, 0],
    )

    # Multiple gradient epochs on the same batch
    anchor_agent = learner_state.anchor_agent  # fixed for inner loop

    def update_epoch(carry: UpdateState, _):
        def loss_fn(agent, metrics):
            return compute_rnad_loss(
                agent=agent,
                anchor_agent=anchor_agent,
                trajectory=trajectory,
                metrics=metrics,
                eta=config.algorithm.eta,
                c_vtrace=config.algorithm.c_vtrace,
                nerd_clip=config.algorithm.nerd_clip,
                nerd_beta=config.algorithm.nerd_beta,
                gamma=config.algorithm.gamma,
            )

        grad = nnx.grad(loss_fn)(carry.agent, carry.metrics)
        carry.optimizer.update(grad)
        return carry, None

    update_carry = UpdateState(
        agent=learner_state.agent,
        optimizer=learner_state.optimizer,
        metrics=learner_state.train_metrics,
        key=learner_state.key,
    )
    update_carry, _ = nnx.scan(update_epoch, length=config.algorithm.num_epoch)(update_carry, None)

    learner_state.agent = update_carry.agent
    learner_state.optimizer = update_carry.optimizer
    learner_state.train_metrics = update_carry.metrics

    return learner_state, None


@partial(nnx.jit, static_argnames=('env', 'config'))
def training_step(
    learner_state: LearnerState,
    env: env_types.BaseEnv,
    config: DictConfig,
) -> LearnerState:
    """Run log_interval single training steps (for efficient JIT compilation)."""
    learner_state, _ = nnx.scan(
        partial(single_training_step, env=env, config=config),
        length=config.logging.log_interval,
    )(learner_state, None)
    return learner_state


def log_metrics(learner_state: LearnerState, logger: BaseLogger, cur_num_update: int):
    train_metrics = learner_state.train_metrics.compute()
    rollout_metrics = learner_state.rollout_metrics.compute()

    logger.log_train_metrics(train_metrics, cur_num_update)

    eps_len = 1 / rollout_metrics['inverse_eps_len']
    ret = rollout_metrics['reward'] / rollout_metrics['inverse_eps_len']
    logger.log_rollout_metrics({'eps_len': eps_len, 'return': ret}, cur_num_update)

    learner_state.train_metrics.reset()
    learner_state.rollout_metrics.reset()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: DictConfig):
    key = jax.random.key(config.seed)

    # Setup env
    env = create_env(config.env.env_name)
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, config.algorithm.num_envs)
    env_state, init_timestep = jax.vmap(env.reset)(init_keys)

    # Setup agent
    key, agent_key = jax.random.split(key)
    agent = create_agent(config.agent.agent_name, key=agent_key)

    # Setup optimizer
    optimizer = nnx.Optimizer(
        agent,
        optax.chain(
            optax.clip_by_global_norm(config.algorithm.max_grad_norm),
            optax.adamw(config.algorithm.lr, eps=1e-5),
        ),
    )

    train_metrics = nnx.MultiMetric(
        loss_v=nnx.metrics.Average("loss_v"),
        loss_nerd=nnx.metrics.Average("loss_nerd"),
        total_loss=nnx.metrics.Average("total_loss"),
    )
    rollout_metrics = nnx.MultiMetric(
        inverse_eps_len=nnx.metrics.Average("inverse_eps_len"),
        reward=nnx.metrics.Average("reward"),
    )

    key, learner_key = jax.random.split(key)
    learner_state = LearnerState(
        key=learner_key,
        env_state=env_state,
        last_timestep=init_timestep,
        agent=agent,
        optimizer=optimizer,
        train_metrics=train_metrics,
        rollout_metrics=rollout_metrics,
        anchor_agent=nnx.clone(agent),
    )

    logger = create_logger(config)
    logger.log_config(config)
    assert config.algorithm.num_inner_update % config.logging.log_interval == 0, \
        "log_interval must divide num_inner_update"

    if config.logging.save_interval > 0:
        learner_state.agent.save_checkpoint(
            Path(config.logging.checkpoint_dir).resolve() / config.run_name, step=0
        )

    total_steps = config.algorithm.num_inner_update * config.algorithm.num_outer_update
    with tqdm(total=total_steps, desc="Training") as pbar:
        for cur_outer in range(config.algorithm.num_outer_update):
            for cur_inner in range(0, config.algorithm.num_inner_update, config.logging.log_interval):
                cur_update = cur_outer * config.algorithm.num_inner_update + cur_inner

                learner_state = training_step(learner_state, env, config)

                cur_update += config.logging.log_interval
                pbar.update(config.logging.log_interval)

                log_metrics(learner_state, logger, cur_update)

                if config.logging.save_interval > 0 and cur_update % config.logging.save_interval == 0:
                    learner_state.agent.save_checkpoint(
                        Path(config.logging.checkpoint_dir).resolve() / config.run_name,
                        step=cur_update,
                    )

            # Update anchor at the end of each outer loop iteration
            learner_state.anchor_agent = nnx.clone(learner_state.agent)

    logger.close()


@hydra.main(version_base=None, config_path="../conf/default", config_name="rnad")
def hydra_main(config: DictConfig) -> None:
    main(config)


if __name__ == '__main__':
    hydra_main()
