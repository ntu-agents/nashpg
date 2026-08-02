from typing import Any, Literal, Tuple
from functools import partial

import chex
import jax
from flax import nnx
import jax.numpy as jnp

from agents import MixtureAgent, BaseAgent
from envs.mytypes import BaseEnv, EnvState, TimeStep


def update_meta_agent(
        meta_agent: MixtureAgent,
        new_agent: BaseAgent,
        slot_idx: int,
        num_active_agents: int,
        key: chex.PRNGKey,
        env: BaseEnv,
        num_steps: int = 1_000,
        num_envs: int = 128,
        solver: Literal['nash', 'uniform'] = 'nash',
        num_solver_iteration: int = 10_000,
) -> None:
    """Add new_agent to the pre-allocated meta-agent and recompute weights in-place.

    Mutates meta_agent in-place — no new object is created, so nnx.jit never
    recompiles due to a shape change.
    """
    capacity = meta_agent.num_agents
    active_agents = [meta_agent.agents[i] for i in range(num_active_agents)]
    all_agents = active_agents + [new_agent]
    total = len(all_agents)

    if solver == 'uniform':
        new_probs_active = jnp.ones(total, dtype=jnp.float32) / total

    elif solver == 'nash':
        payoff_matrix = jnp.zeros((total, total), dtype=jnp.float32)
        for i in range(total):
            for j in range(i + 1, total):
                key, subkey = jax.random.split(key)
                avg_return = compute_avg_return(
                    all_agents[i], all_agents[j], subkey, env, num_envs, num_steps
                )
                payoff_matrix = payoff_matrix.at[i, j].set(avg_return)
                payoff_matrix = payoff_matrix.at[j, i].set(-avg_return)

        new_probs_active = compute_meta_strategy(payoff_matrix, num_iteration=num_solver_iteration)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    padded_probs = jnp.zeros(capacity, dtype=jnp.float32).at[:total].set(new_probs_active)

    meta_agent.update_slot(slot_idx, new_agent)
    meta_agent.update_mixture_probs(padded_probs)


def compute_avg_return(
    agent_i: BaseAgent,
    agent_j: BaseAgent,
    key: chex.PRNGKey,
    env: BaseEnv,
    num_envs: int,
    num_steps: int,
) -> chex.Numeric:
    """Calculate the expected payoff of agent_i playing against agent_j."""
    graphdef, state_i = nnx.split(agent_i)
    _, state_j = nnx.split(agent_j)
    return _compute_avg_return(graphdef, state_i, state_j, key, env, num_envs, num_steps)


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def _compute_avg_return(
    graphdef,
    state_i,
    state_j,
    key: chex.PRNGKey,
    env: BaseEnv,
    num_envs: int,
    num_steps: int,
) -> chex.Numeric:
    agent_i = nnx.merge(graphdef, state_i)
    agent_j = nnx.merge(graphdef, state_j)

    def collect_one_env_step(carry: Tuple[TimeStep, EnvState, chex.PRNGKey, chex.Numeric, chex.Numeric, chex.Numeric], _: Any):
        last_timestep, env_state, key, acc_reward, total_return, num_eps = carry
        key, act_key = jax.random.split(key)

        last_timestep = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), last_timestep)
        action_i = agent_i.get_action(last_timestep.observation, act_key, last_timestep.action_mask)[0]
        action_j = agent_j.get_action(last_timestep.observation, act_key, last_timestep.action_mask)[0]
        last_timestep = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), last_timestep)

        action = jnp.where(last_timestep.current_player == 0, action_i, action_j)
        env_state, new_timestep = env.step(env_state, action)

        acc_reward += new_timestep.reward[0]
        total_return = jnp.where(new_timestep.done, total_return + acc_reward, total_return)
        num_eps = jnp.where(new_timestep.done, num_eps + 1, num_eps)
        acc_reward = jnp.where(new_timestep.done, 0, acc_reward)

        return (new_timestep, env_state, key, acc_reward, total_return, num_eps), None

    def run_single_env(carry):
        final_carry, _ = jax.lax.scan(collect_one_env_step, carry, None, length=num_steps)
        return final_carry

    key1, key2 = jax.random.split(key)
    env_states, timesteps = jax.vmap(env.reset)(jax.random.split(key1, num_envs))
    zeros = jnp.zeros((num_envs,), dtype=jnp.float32)

    _, _, _, _, total_return, num_eps = jax.vmap(run_single_env)(
        (timesteps, env_states, jax.random.split(key2, num_envs), zeros, zeros, zeros)
    )

    return jnp.mean(total_return / num_eps)


@partial(jax.jit, static_argnums=(1,))
def compute_meta_strategy(payoff_matrix: chex.Array, num_iteration: int = 1000) -> chex.Array:
    """Compute player 0's meta-strategy via fictitious play (compiled with lax.scan)."""
    chex.assert_rank(payoff_matrix, 2)
    num_actions_p0, num_actions_p1 = payoff_matrix.shape

    def step(carry, _):
        b0, b1, cum = carry
        p1_mixed = b0 / jnp.sum(b0)
        p0_mixed = b1 / jnp.sum(b1)
        best_p0 = jnp.argmax(payoff_matrix @ p1_mixed)
        best_p1 = jnp.argmax(-payoff_matrix.T @ p0_mixed)
        b0 = b0.at[best_p1].add(1)
        b1 = b1.at[best_p0].add(1)
        cum = cum.at[best_p0].add(1)
        return (b0, b1, cum), None

    init = (jnp.ones(num_actions_p1), jnp.ones(num_actions_p0), jnp.zeros(num_actions_p0))
    (_, _, cum), _ = jax.lax.scan(step, init, None, length=num_iteration)
    return cum / num_iteration
