from typing import Tuple
from functools import partial
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
import jax.numpy as jnp
import jax
import chex
import distrax

from agents.blocks.transformer import TransformerBlock
from envs.liar_dice import MAX_HISTORY_LENGTH

EMB_SIZE = 64
MLP_DIM_SIZE = 64
NUM_BLOCKS = 2

class FeatureExtractor(nnx.Module):
    """Shared module for processing bid history through embeddings and transformer blocks."""
    
    def __init__(self, key: chex.PRNGKey):
        rngs = nnx.Rngs(key)
        
        # Embedding layers
        self.player_embedding = nnx.Embed(3, EMB_SIZE, rngs=rngs)  # Maps -1,0,1 to embeddings
        self.quantity_embedding = nnx.Embed(13, EMB_SIZE, rngs=rngs)  # -1 to 10 (12 values: -1,1,2,...,10)
        self.face_embedding = nnx.Embed(9, EMB_SIZE, rngs=rngs)  # -1 to 6 (8 values: -1,1,2,3,4,5,6)
        self.dice_cnt_embedding = nnx.Linear(6, EMB_SIZE, rngs=rngs)
        self.pos_emb = nnx.Embed(MAX_HISTORY_LENGTH, EMB_SIZE, rngs=rngs)
        
        # Encoder blocks
        self.encoder_blocks = nnx.Sequential(*[
            TransformerBlock(features=EMB_SIZE, num_heads=1, mlp_dim=MLP_DIM_SIZE, rngs=rngs)
            for _ in range(NUM_BLOCKS)
        ])
    
    def __call__(self, bid_history_player, bid_history_quantity, bid_history_face, own_dice_counts):
        """Process bid history through embeddings and transformer blocks."""
        # Shift indices to handle -1 padding
        player_indices = (bid_history_player + 1).astype(jnp.int32)  # -1,0,1 -> 0,1,2
        quantity_indices = (bid_history_quantity + 1).astype(jnp.int32)  # -1,1-10 -> 0,2-11
        face_indices = (bid_history_face + 1).astype(jnp.int32)  # -1,1-6 -> 0,2-7

        # normalize dice counts
        own_dice_counts = own_dice_counts.astype(jnp.float32) / 5.0 # at most have count of 5
        
        # Embed each component
        player_emb = self.player_embedding(player_indices)  # (batch, MAX_HISTORY_LENGTH, EMB_SIZE)
        quantity_emb = self.quantity_embedding(quantity_indices)  # (batch, MAX_HISTORY_LENGTH, EMB_SIZE)
        face_emb = self.face_embedding(face_indices)  # (batch, MAX_HISTORY_LENGTH, EMB_SIZE)
        
        # Process dice counts
        dice_cnt_features = self.dice_cnt_embedding(own_dice_counts)  # (batch, EMB_SIZE)
        dice_cnt_emb = dice_cnt_features[:, None, :]  # (batch, 1, EMB_SIZE) - broadcasts during addition
        
        # Add positional embeddings
        positions = jnp.arange(MAX_HISTORY_LENGTH)  # (MAX_HISTORY_LENGTH,)
        pos_emb = self.pos_emb(positions)  # (MAX_HISTORY_LENGTH, EMB_SIZE)
        
        # fusion embeddings
        hist_emb = player_emb + quantity_emb + face_emb + dice_cnt_emb + pos_emb  # (batch, MAX_HISTORY_LENGTH, EMB_SIZE)
        
        # Apply transformer blocks
        features = self.encoder_blocks(hist_emb)  # (batch, MAX_HISTORY_LENGTH, EMB_SIZE)
        features = jnp.mean(features, axis=1) # (batch, EMB_SIZE)
        
        return features


class LiarDiceAgent(BaseAgent):
    """Agent for Liar's Dice using transformer blocks with separate policy and critic networks."""
    
    def __init__(self, key: chex.PRNGKey):
        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)
        
        # Separate bid history processors for policy and critic
        self.policy_processor = FeatureExtractor(key1)
        self.critic_processor = FeatureExtractor(key2)
        
        # Policy head
        self._policy_head = nnx.Linear(MLP_DIM_SIZE, 61, rngs=rngs)
        self.policy_layers = nnx.Sequential(
            nnx.Linear(EMB_SIZE, MLP_DIM_SIZE, rngs=rngs),
            nnx.gelu,
            self._policy_head,
        )

        # Critic head
        self.critic_layers = nnx.Sequential(
            nnx.Linear(EMB_SIZE, MLP_DIM_SIZE, rngs=rngs),
            nnx.gelu,
            nnx.Linear(MLP_DIM_SIZE, 1, rngs=rngs),
        )
        
        # Initialize all layers
        layer_init(self, rngs.param())
        # Initialize policy head with smaller std
        layer_init(self._policy_head, rngs.param(), std=0.01)
    
    @partial(jax.jit, static_argnums=0)
    def get_value(self, observations: dict) -> chex.Array:
        """Compute state value using the critic network."""
        # Extract components from observation dict
        bid_history_player = observations['bid_history_player']
        bid_history_quantity = observations['bid_history_quantity']
        bid_history_face = observations['bid_history_face']
        own_dice_counts = observations['own_dice_counts']
        
        # Process bid history through critic conv layers
        features = self.critic_processor(
            bid_history_player, bid_history_quantity, bid_history_face, own_dice_counts
        )  # (batch, EMB_SIZE)
        
        # Get value estimate
        value = self.critic_layers(features).squeeze(-1)
        
        return value
    
    @partial(jax.jit, static_argnums=0)
    def get_action(self, observations: dict, key: chex.PRNGKey, action_masks: chex.Array = None) -> chex.Array:
        """Sample action from policy network."""
        dist = self.get_action_distribution(observations, action_masks)
        actions = dist.sample(seed=key)
        return actions
    
    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
        self, observations: dict, key: chex.PRNGKey, action_masks: chex.Array = None
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously."""
        # Extract components
        bid_history_player = observations['bid_history_player']
        bid_history_quantity = observations['bid_history_quantity']
        bid_history_face = observations['bid_history_face']
        own_dice_counts = observations['own_dice_counts']
        
        # Critic processing
        critic_features = self.critic_processor(
            bid_history_player, bid_history_quantity, bid_history_face, own_dice_counts
        )
        values = self.critic_layers(critic_features).squeeze(-1)
        
        # Get action distribution and sample
        dist = self.get_action_distribution(observations, action_masks)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        
        return actions, log_probs, values
    
    @partial(jax.jit, static_argnums=0)
    def get_action_distribution(
        self, observations: dict, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network."""
        # Extract components
        bid_history_player = observations['bid_history_player']
        bid_history_quantity = observations['bid_history_quantity']
        bid_history_face = observations['bid_history_face']
        own_dice_counts = observations['own_dice_counts']
        
        # Process bid history through policy conv layers
        features = self.policy_processor(
            bid_history_player, bid_history_quantity, bid_history_face, own_dice_counts
        )
        
        # Get action logits
        logits = self.policy_layers(features) 
        
        # Apply action mask if provided
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        
        return distrax.Categorical(logits=logits)

