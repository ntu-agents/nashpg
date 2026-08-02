from typing import List, Tuple
from flax import nnx
import chex
import distrax
import jax.numpy as jnp
import jax

import envs.mytypes as env_types
import envs.myspaces as env_spaces
from agents import BaseAgent

class RandomAgent(BaseAgent):

    def __init__(self, key: chex.PRNGKey, env: env_types.BaseEnv):
        
        assert isinstance(env.action_space, env_spaces.Discrete), "only support discrete action"

        self.num_action = env.action_space.n


    def get_value(self, observations: env_types.Observation) -> chex.Array:
        batch_size = jax.tree.leaves(observations)[0].shape[0]
        return jnp.zeros((batch_size, ), dtype=jnp.float32)
    
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None) -> env_types.Action:
        return self.get_action_distribution(observations, action_masks).sample(key=key)

    
    def get_action_and_value(
            self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None
        ) -> Tuple[env_types.Action, chex.Array, chex.Array]:

        dist = self.get_action_distribution(observations, action_masks)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        values = self.get_value(observations)

        return actions, log_probs, values
    
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        batch_size = jax.tree.leaves(observations)[0].shape[0]

        logits = jnp.zeros((batch_size, self.num_action), dtype=jnp.float32)
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        
        return distrax.Categorical(logits=logits)
