"""Head-Up Poker agent (transformer-based)."""

from typing import Tuple
from functools import partial
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
from agents.blocks.transformer import TransformerBlock
import jax.numpy as jnp
import jax
import chex
import distrax

import envs.mytypes as env_types
from envs.head_up_poker import MAX_HISTORY_LENGTH, STARTING_STACK, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CARD_EMB_SIZE  = 64    # card token embedding dimension
CARD_TYPE_DIM  = 64    # card-type (hole vs board) embedding dimension
CARD_HEADS     = 4
CARD_MLP_DIM   = 128

HIST_TOKEN_DIM = 64    # history token dimension (4 parts × 16)
HIST_HEADS     = 4
HIST_MLP_DIM   = 128

META_DIM       = 128   # meta-feature embedding dimension
SHARED_DIM     = 512   # feature extractor output dimension

NUM_CARDS      = 7     # 2 hole + 5 board
NUM_BLOCKS     = 2     # transformer layers in each encoder
_HP = HIST_TOKEN_DIM // 4   # = 16


class FeatureExtractor(nnx.Module):
    """
    Encodes a Head-Up Poker observation into a fixed-size feature vector.

    Cards (7 tokens): Embed + type embed → TransformerBlock → mean pool → 64-dim
    History (L tokens): per-token embed × 4 fields → TransformerBlock (masked) → masked mean pool → 64-dim
    Meta (pot, stacks, bets, street): Linear + street embed → 128-dim
    Fusion: concat(64, 64, 128) → 3-layer MLP → 512-dim
    """

    def __init__(self, key: chex.PRNGKey):
        rngs = nnx.Rngs(key)

        # Card encoder (NUM_BLOCKS layers, no mask needed → Sequential)
        self.card_embedding      = nnx.Embed(53, CARD_EMB_SIZE, rngs=rngs)
        self.card_type_embedding = nnx.Embed(2,  CARD_TYPE_DIM, rngs=rngs)
        self.card_encoder = nnx.Sequential(*[
            TransformerBlock(features=CARD_EMB_SIZE, num_heads=CARD_HEADS, mlp_dim=CARD_MLP_DIM, rngs=rngs)
            for _ in range(NUM_BLOCKS)
        ])

        # History encoder (NUM_BLOCKS layers, mask must be forwarded → named attrs)
        self.hist_player_emb = nnx.Embed(3,  _HP, rngs=rngs)
        self.hist_street_emb = nnx.Embed(5,  _HP, rngs=rngs)
        self.hist_action_emb = nnx.Embed(17, _HP, rngs=rngs)
        self.hist_money_proj = nnx.Linear(1, _HP, rngs=rngs)
        self.hist_encoders = [
            TransformerBlock(features=HIST_TOKEN_DIM, num_heads=HIST_HEADS, mlp_dim=HIST_MLP_DIM, rngs=rngs)
            for _ in range(NUM_BLOCKS)
        ]

        # Meta
        self.meta_proj        = nnx.Linear(5, META_DIM, rngs=rngs)
        self.street_embedding = nnx.Embed(5, META_DIM, rngs=rngs)

        # Fusion MLP
        combined_dim = CARD_EMB_SIZE + HIST_TOKEN_DIM + META_DIM
        self.fusion = nnx.Sequential(
            nnx.Linear(combined_dim, SHARED_DIM, rngs=rngs), nnx.gelu,
            nnx.Linear(SHARED_DIM,  SHARED_DIM, rngs=rngs), nnx.gelu,
            nnx.Linear(SHARED_DIM,  SHARED_DIM, rngs=rngs), nnx.gelu,
        )

    def __call__(self, obs: env_types.Observation) -> chex.Array:
        hole_cards     = obs['hole_cards']       # (B, 2)
        board_cards    = obs['board_cards']      # (B, 5)
        pot_size       = obs['pot_size']         # (B,)
        street         = obs['street']           # (B,)
        stacks         = obs['stacks']           # (B, 2)
        bets           = obs['bets']             # (B, 2)
        action_history = obs['action_history']   # (B, L, 4)

        B = hole_cards.shape[0]

        # --- Card encoder ---
        all_cards = jnp.concatenate([hole_cards + 1, board_cards + 1], axis=1)  # (B, 7)
        card_types = jnp.concatenate([
            jnp.zeros((B, 2), dtype=jnp.int32),
            jnp.ones( (B, 5), dtype=jnp.int32),
        ], axis=1)                                                               # (B, 7)
        card_tokens = (
            self.card_embedding(all_cards) + self.card_type_embedding(card_types)
        )                                                                        # (B, 7, CE)
        card_tokens = self.card_encoder(card_tokens)
        card_feat   = card_tokens.mean(axis=1)                                  # (B, CE)

        # --- History encoder ---
        valid       = (action_history[:, :, 0] >= 0)                            # (B, L)
        h_player    = self.hist_player_emb(action_history[:, :, 0] + 1)
        h_street    = self.hist_street_emb(jnp.maximum(action_history[:, :, 1], 0))
        h_action    = self.hist_action_emb(action_history[:, :, 2] + 1)
        h_money     = self.hist_money_proj(
            jnp.maximum(action_history[:, :, 3:4], 0).astype(jnp.float32) / float(STARTING_STACK)
        )
        hist_tokens = jnp.concatenate([h_player, h_street, h_action, h_money], axis=-1)  # (B, L, HD)
        attn_mask   = valid[:, None, None, :]
        for block in self.hist_encoders:
            hist_tokens = block(hist_tokens, mask=attn_mask)
        vmask       = valid[..., None].astype(jnp.float32)
        hist_feat   = (hist_tokens * vmask).sum(axis=1) / jnp.maximum(vmask.sum(axis=1), 1.0)  # (B, HD)

        # --- Meta ---
        norm      = lambda x: x.astype(jnp.float32) / float(STARTING_STACK)
        meta_vec  = jnp.concatenate([norm(pot_size)[..., None], norm(stacks), norm(bets)], axis=-1)
        meta_feat = self.meta_proj(meta_vec) + self.street_embedding(street)    # (B, MD)

        combined = jnp.concatenate([card_feat, hist_feat, meta_feat], axis=-1)
        return self.fusion(combined)


class HeadUpPokerAgent(BaseAgent):
    """Head-Up Poker agent using transformers for card and action history encoding."""

    def __init__(self, key: chex.PRNGKey):
        key, k1, k2, k3 = jax.random.split(key, 4)
        rngs = nnx.Rngs(k3)

        self.policy_feature_extractor = FeatureExtractor(k1)
        self.critic_feature_extractor = FeatureExtractor(k2)

        self.policy_head = nnx.Linear(SHARED_DIM, NUM_ACTIONS, rngs=rngs)
        self.critic_head = nnx.Linear(SHARED_DIM, 1, rngs=rngs)

        layer_init(self, rngs.param())
        layer_init(self.policy_head, rngs.param(), std=0.01)

    @partial(jax.jit, static_argnums=0)
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        return self.critic_head(self.critic_feature_extractor(observations)).squeeze(-1)

    @partial(jax.jit, static_argnums=0)
    def get_action(
        self,
        observations: env_types.Observation,
        key: chex.PRNGKey,
        action_masks: chex.Array = None,
    ) -> chex.Array:
        return self.get_action_distribution(observations, action_masks).sample(seed=key)

    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
        self,
        observations: env_types.Observation,
        key: chex.PRNGKey,
        action_masks: chex.Array = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        logits = self.policy_head(self.policy_feature_extractor(observations))
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        dist = distrax.Categorical(logits=logits)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        values = self.critic_head(self.critic_feature_extractor(observations)).squeeze(-1)
        return actions, log_probs, values

    @partial(jax.jit, static_argnums=0)
    def get_action_distribution(
        self,
        observations: env_types.Observation,
        action_masks: chex.Array = None,
    ) -> distrax.Distribution:
        logits = self.policy_head(self.policy_feature_extractor(observations))
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        return distrax.Categorical(logits=logits)
