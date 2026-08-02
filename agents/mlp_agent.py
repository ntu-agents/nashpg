from typing import Tuple
from functools import partial
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
import jax
import jax.numpy as jnp
import chex
import distrax

MLP_DIM = 512

class FeatureExtractor(nnx.Module):
    
    def __init__(self, key: chex.PRNGKey, input_dim: int):
        rngs = nnx.Rngs(key)
        self.mlp = nnx.Sequential(
            nnx.Linear(in_features=input_dim, out_features=MLP_DIM, rngs=rngs),
            nnx.gelu,
            nnx.Linear(in_features=MLP_DIM, out_features=MLP_DIM, rngs=rngs),
            nnx.gelu,
            nnx.Linear(in_features=MLP_DIM, out_features=MLP_DIM, rngs=rngs),
            nnx.gelu
        )
        
    def __call__(self, observations: chex.Array) -> chex.Array:
        # Flatten input 
        flattened = observations.reshape(observations.shape[0], -1).astype(jnp.float32)
        return self.mlp(flattened)

class MLPAgent(BaseAgent):
    
    def __init__(self, key: chex.PRNGKey, input_dim: int, output_dim: int):
        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)
        
        # Separate feature extractors for policy and critic (no parameter sharing)
        self.policy_extractor = FeatureExtractor(key1, input_dim)
        self.critic_extractor = FeatureExtractor(key2, input_dim)
        
        # Policy and critic heads
        self._policy_head = nnx.Linear(in_features=MLP_DIM, out_features=output_dim, rngs=rngs)
        self._critic_head = nnx.Linear(in_features=MLP_DIM, out_features=1, rngs=rngs)

        # Initialize modules
        layer_init(self, nnx.Rngs(key).param())
        layer_init(self._policy_head, nnx.Rngs(key).param(), std=0.01)
        
    @partial(jax.jit, static_argnums=0)
    def get_value(self, observations: chex.Array) -> chex.Array:
        """Compute state value."""
        return self._critic_head(self.critic_extractor(observations)).squeeze(-1)
    
    @partial(jax.jit, static_argnums=0)
    def get_action(self, observations: chex.Array, key: chex.PRNGKey, action_masks: chex.Array = None) -> chex.Array:
        """Sample action from policy."""
        return self.get_action_distribution(observations, action_masks).sample(seed=key)
    
    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
            self, observations: chex.Array, key: chex.PRNGKey, action_masks: chex.Array = None
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Sample action and compute log probability and value."""
        logits = self._policy_head(self.policy_extractor(observations))
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        
        dist = distrax.Categorical(logits=logits)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        values = self._critic_head(self.critic_extractor(observations)).squeeze(-1)
        
        return actions, log_probs, values
    
    @partial(jax.jit, static_argnums=0)
    def get_action_distribution(
        self, observations: chex.Array, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network."""
        logits = self._policy_head(self.policy_extractor(observations))
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        return distrax.Categorical(logits=logits)