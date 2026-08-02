from functools import partial
from typing import Any, Tuple, Union, List

import jax
import jax.numpy as jnp
from flax import nnx
import chex

import train.mytypes as train_types
import envs.mytypes as env_types
from agents import BaseAgent, MixtureAgent

@partial(nnx.jit, static_argnames=('env', 'num_envs', 'num_steps'))
def collect_and_process_trajectories(
    env: env_types.BaseEnv,
    agent: Union[BaseAgent, List[BaseAgent]],
    env_state: env_types.EnvState,
    last_timestep: env_types.TimeStep,
    metrics: nnx.MultiMetric,
    key: chex.PRNGKey,
    num_envs: int,
    num_steps: int,
    gamma: float,
    gae_gamma: float,
    active_agent_idx: Union[chex.Array, None] = None,
) -> Union[Tuple[env_types.EnvState, env_types.TimeStep, nnx.MultiMetric, train_types.Dataset],
           Tuple[env_types.EnvState, env_types.TimeStep, nnx.MultiMetric, train_types.Dataset, chex.Array]]:
    """
    Collect trajectories and process them into a training dataset.
    
    Args:
        env: Environment instance
        agent: Single agent or list of agents (length must match num_agents)
        env_state: Current environment state with shape (num_envs, ...)
        last_timestep: Last timestep from previous rollout with shape (num_envs, ...)
        key: JAX random key (scalar)
        num_envs: Number of parallel environments
        num_steps: Number of steps to collect
        gamma: Discount factor for GAE (0 < gamma <= 1)
        gae_gamma: GAE lambda parameter for bias-variance tradeoff (0 < gae_gamma <= 1)
        
    Returns:
        Tuple of (new_env_state, new_last_timestep, dataset)
        - new_env_state: Updated environment states, shape (num_envs, ...)
        - new_last_timestep: Final timestep from rollout, shape (num_envs, ...)
        - new_metrics: Update metrics hold the rollout metrics
        - dataset: Training dataset with flattened batch dimension (num_envs * num_steps, ...)
    """
    # Validate agent input
    num_agents = env.num_agents
    if isinstance(agent, list):
        assert len(agent) == num_agents, f"Agent list length {len(agent)} must match num_agents {num_agents}"
    
    # Collect trajectories
    if active_agent_idx is not None:
        env_state, last_timestep, transitions, active_agent_idx = collect_trajectories(
            env, agent, env_state, last_timestep, key, num_envs, num_steps, active_agent_idx=active_agent_idx
        )
    else:
        env_state, last_timestep, transitions = collect_trajectories(
            env, agent, env_state, last_timestep, key, num_envs, num_steps
        )
    transitions: train_types.Transition = transitions
    
    # Calculate GAE advantages and target values
    advantages, target_values, valid_mask = calculate_gae(
        transitions, num_agents, gamma, gae_gamma
    )

    # Flatten to batch dimension (num_envs * num_steps, ...)
    batch_size = num_envs * num_steps
    
    # Create dataset
    dataset = train_types.Dataset(
        action=transitions.action.reshape(batch_size, *transitions.action.shape[2:]),
        value=transitions.value.reshape(batch_size),
        log_prob=transitions.log_prob.reshape(batch_size),
        observation=jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[2:]), transitions.observation),
        action_mask=transitions.action_mask.reshape(batch_size, *transitions.action_mask.shape[2:]),
        current_player=transitions.current_player.reshape(batch_size),
        advantage=advantages.reshape(batch_size),
        target_value=target_values.reshape(batch_size),
        valid_mask=valid_mask.reshape(batch_size),
    )

    # log avg episode length
    metrics.update(
        inverse_eps_len=transitions.is_new_eps.reshape(batch_size), # ratio of new episode
        reward=transitions.reward.reshape(batch_size, num_agents)[:, 0] # player 0 rewards
    )
    
    if active_agent_idx is not None:
        return env_state, last_timestep, metrics, dataset, active_agent_idx
    else:
        return env_state, last_timestep, metrics, dataset

@partial(nnx.jit, static_argnames=('env', 'num_envs', 'num_steps'))
def collect_trajectories(
    env: env_types.BaseEnv,
    agent: Union[BaseAgent, List[BaseAgent]],
    env_state: env_types.EnvState,
    last_timestep: env_types.TimeStep,
    key: chex.PRNGKey,
    num_envs: int,
    num_steps: int,
    active_agent_idx: Union[chex.Array, None] = None,
) -> Union[Tuple[env_types.EnvState, env_types.TimeStep, train_types.Transition],
           Tuple[env_types.EnvState, env_types.TimeStep, train_types.Transition, chex.Array]]:
    """
    Collect trajectories from multiple environments for a specified number of steps.
    
    This function performs rollouts across multiple environments, collecting transitions
    for training. Each environment is stepped forward for num_steps.
    
    Args:
        env: The environment instance implementing BaseEnv interface
        agent: Single agent or list of agents for multi-agent environments
        env_state: Current state of all environments, shape (num_envs, ...)
        last_timestep: The last timestep from previous rollout, shape (num_envs, ...)
        key: JAX random key for action sampling (single key, will be split internally)
        num_envs: Number of parallel environments to run
        num_steps: Number of steps to collect from each environment
        
    Returns:
        Tuple containing:
        - env_state: Updated environment states after rollout, shape (num_envs, ...)
        - last_timestep: Final timestep from rollout, shape (num_envs, ...)
        - transitions: Collected transitions, shape (num_envs, num_steps, ...)
                      Contains action, value, reward, log_prob, observation, 
                      action_mask, current_player, and is_new_eps fields
                      
    Note:
        - Environments are assumed to auto-reset when done=True
        - All inputs must have consistent batch dimensions of num_envs
    """

    chex.assert_equal_shape([last_timestep.done, env_state.done])
    chex.assert_shape(last_timestep.done, (num_envs, )) # should input as a batch with (num_envs, ...)
    chex.assert_rank(key, 0) # one key

    def collect_one_env_step(carry: Tuple[env_types.TimeStep, env_types.EnvState, chex.PRNGKey, chex.Array] , _: Any):
        """step env for a single env, i.e., no batch dimesions"""

        if active_agent_idx is None:
            last_timestep, env_state, key = carry
            current_active_agent_idx = None
        else:
            last_timestep, env_state, key, current_active_agent_idx = carry
        key, act_key = jax.random.split(key)

        # add batch dimesion for input to agent
        last_timestep: env_types.TimeStep = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), last_timestep)
        
        # agent act
        if isinstance(agent, list):
            # Run all agents and pick the output from current player
            all_agent_outputs = []
            for agent_i in agent:
                if isinstance(agent_i, MixtureAgent):
                    # This is a MixtureAgent, use specific agent index
                    agent_output_i = agent_i.get_action_and_value_by_index(
                        last_timestep.observation, act_key, last_timestep.action_mask, 
                        current_active_agent_idx
                    )
                else:
                    agent_output_i = agent_i.get_action_and_value(
                        last_timestep.observation, act_key, last_timestep.action_mask
                    )
                all_agent_outputs.append(agent_output_i)
            
            # Stack outputs from all agents and select the current player's output
            current_player = jnp.squeeze(last_timestep.current_player, axis=0)
            stacked_outputs = jax.tree.map(lambda *outputs: jnp.stack(outputs, axis=0), *all_agent_outputs)
            agent_output = jax.tree.map(lambda x: x[current_player], stacked_outputs)
        else:
            agent_output = agent.get_action_and_value(
                last_timestep.observation, act_key, last_timestep.action_mask
            )

        # remove batch dimesion
        action, log_prob, value = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), agent_output)
        
        # Remove batch dimension from last_timestep after agent processing
        last_timestep = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), last_timestep)

        # step env
        env_state, new_timestep = env.step(env_state, action)
        
        # Handle resampling for MixtureAgent when episode ends
        if isinstance(agent, list):
            for agent_i in agent:
                if isinstance(agent_i, MixtureAgent):
                    current_active_agent_idx, key = agent_i.maybe_resample(
                        current_active_agent_idx, new_timestep.done, key
                    )

        # create new transition
        transition = train_types.Transition(
            is_new_eps=last_timestep.done, # our env atuto reset at Same mode, so when last step is done, meaning this step is new episode
            action=action,
            value=value,
            reward=new_timestep.reward,
            log_prob=log_prob,
            observation=last_timestep.observation,
            action_mask=last_timestep.action_mask,
            current_player=last_timestep.current_player
        )

        # return next carry and this transition
        if active_agent_idx is None:
            return (new_timestep, env_state, key), transition    
        else:
            return (new_timestep, env_state, key, current_active_agent_idx), transition
    
    # the output of rollout will have extra dimesion of (num_steps, ...)
    single_env_rollout_fc = nnx.scan(
        collect_one_env_step, length=num_steps
    )

    # batch for num_envs
    batch_env_rollout_fc = nnx.vmap(single_env_rollout_fc, in_axes=(0, None), out_axes=0)
    
    # prepare batched keys
    keys = jax.random.split(key, num_envs)

    # perform batch rollout, return with shape (num_envs, num_steps, ...)
    if active_agent_idx is None:
        (last_timestep, env_state, _), transitions = batch_env_rollout_fc(
            (last_timestep, env_state, keys), # carry
            None # empty Ys
        )
    else:
        (last_timestep, env_state, _, active_agent_idx), transitions = batch_env_rollout_fc(
            (last_timestep, env_state, keys, active_agent_idx), # carry
            None # empty Ys
        )

    chex.assert_shape(transitions.is_new_eps, (num_envs, num_steps))

    if active_agent_idx is None:
        return env_state, last_timestep, transitions
    else:
        return env_state, last_timestep, transitions, active_agent_idx


@partial(nnx.jit, static_argnames=('num_agents', ))
def calculate_gae(
    transitions: train_types.Transition,
    num_agents: int, 
    gamma: float, 
    gae_gamma: float
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Calculate Generalized Advantage Estimation (GAE) for multi-agent trajectories.
    
    Args:
        transitions: Trajectory data with shape (num_envs, num_steps, ...)
        num_agents: Number of agents in the environment
        gamma: Discount factor for future rewards
        gae_gamma: GAE lambda parameter for bias-variance tradeoff
        
    Returns:
        Tuple of (advantages, target_values, valid_mask) with shape (num_envs, num_steps)
        - advantages: GAE advantages for each transition
        - target_values: Target values for critic training
        - valid_mask: Boolean mask indicating which advantages are valid
    """

    def calculate_one_env_gae(
        carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array], 
        transition: train_types.Transition
    ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array], Tuple[chex.Array, chex.Array, chex.Array]]:
        """
        Calculate GAE for a single environment step (reverse scan).
        
        Args:
            carry: Tuple of (gae, next_value, reward_accum, has_next_value, is_valid) arrays, each shape (num_agents,) beside is_valid shape ()
            transition: Current transition (earlier in time due to reverse scan)
            
        Returns:
            Tuple of (new_carry, (advantage, target_value, valid_mask)) where:
            - new_carry: Updated carry for previous step
            - advantage: GAE advantage for current player (scalar) 
            - target_value: Target value for current player (scalar)
            - valid_mask: Whether this advantage is valid (scalar)
        """
        next_gae, next_value, reward_accum, has_next_value, next_is_new_eps, next_is_valid = carry
        
        # Extract transition data
        current_player = transition.current_player  # scalar
        reward = transition.reward  # shape: (num_agents,)
        value = transition.value  # scalar value for current player
        is_new_eps = transition.is_new_eps  # scalar
        
        # Reset states at episode boundaries
        next_gae = jnp.where(next_is_new_eps, jnp.zeros_like(next_gae), next_gae)
        reward_accum = jnp.where(next_is_new_eps, jnp.zeros_like(reward_accum), reward_accum)
        has_next_value = jnp.where(next_is_new_eps, jnp.zeros_like(has_next_value), has_next_value)
        next_value = jnp.where(next_is_new_eps, jnp.zeros_like(next_value), next_value)
        
        # Accumulate rewards for all agents
        reward_accum = reward_accum + reward
        
        # Current player consumes their accumulated reward
        player_reward = reward_accum[current_player]
        
        # Reset reward accumulator for current player after consumption
        consumed_reward_accum = reward_accum.at[current_player].set(0.0)
        
        # Check if we have valid next_value for current player
        player_has_next = has_next_value[current_player]
        
        # Calculate TD error for current player using consumed reward
        td_error = player_reward + gamma * next_value[current_player] - value
        
        # Update GAE for current player only
        new_gae_current = td_error + gamma * gae_gamma * next_gae[current_player]
        new_gae = next_gae.at[current_player].set(new_gae_current)
        
        # Calculate advantage and target value for current player
        # Only valid if 1) have next value OR 2) is complete episode
        is_valid = player_has_next | next_is_new_eps | next_is_valid[current_player]
        advantage = jnp.where(is_valid, new_gae_current, 0.0)
        target_value = jnp.where(is_valid, advantage + value, value)
        valid_mask = is_valid
        
        # Update next_value and has_next_value for current player
        updated_next_value = next_value.at[current_player].set(value)
        updated_has_next_value = has_next_value.at[current_player].set(True)

        # update next is value
        updated_next_is_valid = next_is_valid.at[current_player].set(is_valid) | next_is_new_eps # is complete episode
        
        return (
            new_gae, updated_next_value, consumed_reward_accum, updated_has_next_value, is_new_eps, updated_next_is_valid
        ), (advantage, target_value, valid_mask)

    # Process each environment in parallel
    def process_single_env(transitions_single_env: train_types.Transition) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Process GAE calculation for a single environment's trajectory."""
        
        # Initialize carry for reverse scan
        init_gae = jnp.zeros(num_agents, dtype=jnp.float32)
        init_next_value = jnp.zeros(num_agents, dtype=jnp.float32)
        init_reward_accum = jnp.zeros(num_agents, dtype=jnp.float32)
        init_has_next_value = jnp.zeros(num_agents, dtype=jnp.bool)
        init_next_is_new_eps = jnp.array(False, dtype=jnp.bool)
        init_next_is_valid = jnp.zeros(num_agents, dtype=jnp.bool)
        init_carry = (init_gae, init_next_value, init_reward_accum, init_has_next_value, init_next_is_new_eps, init_next_is_valid)
        
        # Use lax.scan in reverse to propagate values backward
        _, (advantages, target_values, valid_mask) = jax.lax.scan(
            calculate_one_env_gae,
            init_carry,
            transitions_single_env,
            reverse=True
        )
        
        return advantages, target_values, valid_mask
    
    # Vectorize over environments
    advantages, target_values, valid_mask = jax.vmap(process_single_env)(transitions)
    
    return advantages, target_values, valid_mask

