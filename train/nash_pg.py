"""
Training script for simultaneous update self-play

Assumption:
* Action space is Discrete
* Action space and Observation space are same for all agents
"""

import os
from pathlib import Path
from typing import Any, Optional, Tuple
import logging
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Suppress verbose Orbax checkpoint logging
logging.getLogger('absl').setLevel(logging.WARNING)

from functools import partial
from tqdm import tqdm
import jax
from flax import nnx
import chex
import optax
import hydra
from omegaconf import DictConfig

from envs import create_env
import envs.mytypes as env_types
from agents import create_agent, BaseAgent
from train.core import collect_and_process_trajectories, update_agent
from train.loggers import create_logger, BaseLogger



@chex.dataclass
class LearnerState:
    key: chex.PRNGKey
    env_state: env_types.EnvState
    last_timestep: env_types.TimeStep
    agent: BaseAgent
    optimizer: nnx.Optimizer
    train_metrics: nnx.MultiMetric
    rollout_metrics: nnx.MultiMetric
    mag_agent: Optional[BaseAgent] # use for regularization


@partial(nnx.jit, static_argnames=('env', 'config'))
def single_training_step(
        learner_state: LearnerState,
        _: Any,
        env: env_types.BaseEnv,
        config: DictConfig
    ) -> Tuple[LearnerState, Any]:
    """
    Single training step for use with nnx.scan
    """
    
    """collect and process trajactories """
    learner_state.key, collect_key = jax.random.split(learner_state.key)
    learner_state.env_state, learner_state.last_timestep, learner_state.rollout_metrics, dataset = collect_and_process_trajectories(
        env = env,
        agent = learner_state.agent,
        env_state = learner_state.env_state,
        last_timestep = learner_state.last_timestep,
        metrics = learner_state.rollout_metrics,
        key = collect_key,
        num_envs = config.algorithm.num_envs,
        num_steps = config.algorithm.num_steps,
        gamma = config.algorithm.gamma,
        gae_gamma = config.algorithm.gae_gamma
    )


    """perform ppo update"""
    learner_state.key, update_key = jax.random.split(learner_state.key)
    learner_state.agent, learner_state.optimizer, learner_state.train_metrics = update_agent(
        agent = learner_state.agent,
        mag_agent = learner_state.mag_agent,
        optimizer = learner_state.optimizer,
        dataset = dataset,
        metrics = learner_state.train_metrics,
        key = update_key,
        ent_coef = config.algorithm.ent_coef,
        mag_coef = config.algorithm.mag_coef,
        mag_divergence_type = config.algorithm.mag_divergence_type,
        clip_eps = config.algorithm.clip_eps,
        num_minibatches = config.algorithm.num_minibatches,
        num_ppo_epoch = config.algorithm.num_ppo_epoch,
        only_use_player0_experience = False,
    )

    return learner_state, None


@partial(nnx.jit, static_argnames=('env', 'config'))
def training_step(
        learner_state: LearnerState,
        env: env_types.BaseEnv,
        config: DictConfig
    ) -> LearnerState:
    """Training step that runs log_interval iterations of single_training_step"""
    learner_state, _ = nnx.scan(
        partial(single_training_step, env=env, config=config),
        length=config.logging.log_interval,
    )(learner_state, None)

    return learner_state


def log_metrics(learner_state: LearnerState, logger: BaseLogger, cur_num_update: int):
    """Log training and rollout metrics"""
    
    train_metrics = learner_state.train_metrics.compute() 
    rollout_metrics = learner_state.rollout_metrics.compute()
    
    # Log train metrics
    logger.log_train_metrics(train_metrics, cur_num_update)

    # Process and log rollout metrics
    eps_len = 1 / rollout_metrics['inverse_eps_len']
    ret = rollout_metrics['reward'] / rollout_metrics['inverse_eps_len']
    processed_rollout_metrics = {
        'eps_len': eps_len,
        'return': ret
    }
    logger.log_rollout_metrics(processed_rollout_metrics, cur_num_update)

    learner_state.train_metrics.reset()
    learner_state.rollout_metrics.reset()


def main(config: DictConfig):
    key = jax.random.key(config.seed)

    # setup env
    env = create_env(config.env.env_name)
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, config.algorithm.num_envs)
    env_state, init_timestep = jax.vmap(env.reset)(init_keys) # (num_envs, )

    # setup agent
    key, agent_key = jax.random.split(key)
    agent = create_agent(config.agent.agent_name, key=agent_key)

    # setup optimizer & metrics
    optimizer = nnx.Optimizer(agent, optax.chain(optax.clip_by_global_norm(config.algorithm.max_grad_norm), optax.adamw(config.algorithm.lr, eps=1e-5)))
    train_metrics = nnx.MultiMetric(
        actor_loss = nnx.metrics.Average("actor_loss"),
        ppo_loss = nnx.metrics.Average("ppo_loss"),
        entropy = nnx.metrics.Average("entropy"),
        critic_loss = nnx.metrics.Average("critic_loss"),
        approx_kl = nnx.metrics.Average("approx_kl"),
        mag_kl = nnx.metrics.Average("mag_kl"),
        clip_frac = nnx.metrics.Average("clip_frac"),
        explained_var = nnx.metrics.Average("explained_var"),
    )
    rollout_metrics = nnx.MultiMetric(
        inverse_eps_len = nnx.metrics.Average("inverse_eps_len"),
        reward = nnx.metrics.Average("reward"),
    )

    # setup learner state
    key, learner_key = jax.random.split(key)
    learner_state = LearnerState(
        key=learner_key,
        env_state=env_state,
        last_timestep=init_timestep,
        agent=agent,
        optimizer=optimizer,
        train_metrics=train_metrics,
        rollout_metrics=rollout_metrics,
        mag_agent=nnx.clone(agent), # init as the same
    )

    # setup logger
    logger = create_logger(config)
    logger.log_config(config)
    assert config.algorithm.num_inner_update % config.logging.log_interval == 0, "log_interval must be a divisible by num_update"

    # save first model
    if config.logging.save_interval > 0:
        learner_state.agent.save_checkpoint(Path(config.logging.checkpoint_dir).resolve() / config.run_name, step=0)
    
    # training loop
    with tqdm(total=config.algorithm.num_inner_update * config.algorithm.num_outer_update, desc="Training") as pbar:
        for cur_num_outer_update in range(0, config.algorithm.num_outer_update):
            for cur_num_inner_update in range(0, config.algorithm.num_inner_update, config.logging.log_interval):
                cur_num_update = cur_num_outer_update * config.algorithm.num_inner_update + cur_num_inner_update
                
                # training step for `log_interval` steps
                learner_state: LearnerState = training_step(learner_state, env, config)

                # update progress bar
                cur_num_update += config.logging.log_interval
                pbar.update(config.logging.log_interval)

                # logging
                log_metrics(learner_state, logger, cur_num_update)

                # save model
                if config.logging.save_interval > 0 and cur_num_update % config.logging.save_interval == 0:
                    learner_state.agent.save_checkpoint(Path(config.logging.checkpoint_dir).resolve() / config.run_name, step=cur_num_update)

            # update magnet
            learner_state.mag_agent = nnx.clone(learner_state.agent)


    # close logger
    logger.close()


@hydra.main(version_base=None, config_path="../conf/default", config_name="nash_pg")
def hydra_main(config: DictConfig) -> None:
    main(config)

if __name__ == '__main__':
    hydra_main()