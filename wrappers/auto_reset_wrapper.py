from typing import Tuple
from functools import partial
import chex
import jax

import envs.mytypes as env_types

class AutoResetWrapper(env_types.BaseEnv):
    """
    Auto reset the env, with MODE=SAME_STEP
    """

    def __init__(self, env: env_types.BaseEnv):
        self._env = env

    @property
    def env_name(self) -> str:
        return self._env.env_name

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def action_space(self) -> env_types.Space:
        return self._env.action_space

    @property
    def observation_space(self) -> env_types.Space:
        return self._env.observation_space

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[env_types.EnvState, env_types.TimeStep]:
        return self._env.reset(key)

    @partial(jax.jit, static_argnums=0)
    def step(self, state: env_types.EnvState, action: env_types.Action) -> Tuple[env_types.EnvState, env_types.TimeStep]:
        state, timestep = self._env.step(state, action)

        state, timestep = jax.lax.cond(
            timestep.done,
            self._auto_reset,
            lambda s, t: (s, t), # not done -> continue as normal
            state, timestep
        )

        return state, timestep
    
    @partial(jax.jit, static_argnums=0)
    def _auto_reset(self, state: env_types.EnvState, timestep: env_types.TimeStep) -> Tuple[env_types.EnvState, env_types.TimeStep]:
        """auto reset with mode=same_step"""
        new_state, new_timestep = self._env.reset(state.key)

        return new_state, env_types.TimeStep(
            reward=timestep.reward,
            done=timestep.done,
            observation=new_timestep.observation,
            action_mask=new_timestep.action_mask,
            current_player=new_timestep.current_player,
            info=timestep.info, # NOTE: we use terminated step info, this mean the new episode first step info is gone
        )