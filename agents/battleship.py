from typing import Tuple
from functools import partial
from flax import nnx
import envs.mytypes as env_types
from agents import BaseAgent
from agents.utils import layer_init
import jax.numpy as jnp
import jax
import chex
import distrax

EMB_SIZE = 64
D_DIM = 128

class FeatureExtractor(nnx.Module):
    def __init__(self, rngs: nnx.Rngs = None):
        # Board embedding: -5 to 1 mapped to 0-6, smaller dimension
        self.board_embedding = nnx.Embed(7, EMB_SIZE, rngs=rngs)
        
        # Stage embedding
        self.stage_embedding = nnx.Embed(2, EMB_SIZE, rngs=rngs)
        
        # Ship projection: 5 my + 5 enemy = 10 inputs
        self.ship_proj = nnx.Linear(10, EMB_SIZE, rngs=rngs)
        
        self.cnn = nnx.Sequential(
            nnx.Conv(EMB_SIZE, D_DIM, (3, 3), strides=1, padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.Conv(D_DIM, D_DIM, (3, 3), strides=1, padding="SAME", rngs=rngs),
            nnx.relu,
        )
        
    def __call__(self, observations: env_types.Observation) -> chex.Array:
        stage = observations['stage']
        board = observations['board']
        my_alive_ships = observations['my_alive_ships']
        enemy_alive_ships = observations['enemy_alive_ships']

        # Map board to embedding indices: [-5,1] -> [0,6]
        board_indices = jnp.clip(board + 5, 0, 6).astype(jnp.int32)
        board_emb = self.board_embedding(board_indices)  # (batch, 10, 10, EMB_SIZE)
        
        # Broadcast stage embedding
        stage_emb = self.stage_embedding(stage)[:, None, None, :]  # (batch, 1, 1, EMB_SIZE)
        
        # Project and broadcast ship features
        ships = jnp.concatenate([my_alive_ships.astype(jnp.float32), enemy_alive_ships.astype(jnp.float32)], axis=-1)
        ships_emb = self.ship_proj(ships)[:, None, None, :]  # (batch, 1, 1, EMB_SIZE)
        
        # Combine embeddings # (batch, 10, 10, EMB_SIZE)
        features = board_emb + stage_emb + ships_emb
        
        # Feature extraction
        features = self.cnn(features) # (batch, 10, 10, D_DIM)
        
        return features

class BattleShipAgent(BaseAgent):
    """Cheaper Battleship agent with unified networks for both stages"""
    
    def __init__(self, key: chex.PRNGKey):
        rngs = nnx.Rngs(key)
        
        self.policy_feature_extractor = FeatureExtractor(rngs=rngs)
        self.critic_feature_extractor = FeatureExtractor(rngs=rngs)

        self.policy_head = nnx.Sequential(
            nnx.Conv(D_DIM, D_DIM, (3, 3), strides=1, padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.Conv(D_DIM, 1, (3, 3), strides=1, padding="SAME", rngs=rngs),
        )

        self.critic_head = nnx.Linear(D_DIM, 1, rngs=rngs)
        
        # Initialize networks
        self._init_networks(rngs.param())
        
    def _init_networks(self, key: chex.PRNGKey):
        """Initialize all network parameters"""
        # Initialize all modules
        key1, key2 = jax.random.split(key)
        layer_init(self, key1)
        
        layer_init(self.policy_head, key2, std=0.01)
    
    
    @partial(jax.jit, static_argnums=0)
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value using the value network"""
        features = self.critic_feature_extractor(observations) # (batch, 10, 10, D_DIM)
        features = jnp.mean(features, axis=(1,2)) # (batch, D_DIM)
        value = self.critic_head(features) # (batch, 1)

        return value.squeeze(-1)  # Remove last dimension to get (batch_size,)
    
    @partial(jax.jit, static_argnums=0)
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None) -> chex.Array:
        """Sample action from policy network"""
        chex.assert_rank(key, 0)  # Key has rank 0
        
        dist = self.get_action_distribution(observations, action_masks)
        actions = dist.sample(seed=key)
        
        return actions
    
    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
        self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously"""
        chex.assert_rank(key, 0)  # Key has rank 0
        
        dist = self.get_action_distribution(observations, action_masks)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        value = self.get_value(observations)
        
        return actions, log_probs, value
    
    @partial(jax.jit, static_argnums=0)
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        """Get action distribution"""
        features = self.policy_feature_extractor(observations) # (batch, 10, 10, D_DIM)
        logits = self.policy_head(features).squeeze(-1) # (batch, 10, 10)
        logits = logits.reshape(logits.shape[0], -1) # (batch, 100)
        
        # Apply action masks
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        
        dist = distrax.Categorical(logits=logits)
        return dist
    