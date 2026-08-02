"""Kuhn poker environment implementation using JAX with perfect recall.

This module provides a JAX-based implementation of Kuhn poker
for multi-agent reinforcement learning. The environment follows the BaseEnv
interface and supports two players in a simplified poker game with perfect
recall observation encoding.

Example:
    >>> env = KuhnPoker()
    >>> key = jax.random.key(42)
    >>> state, timestep = env.reset(key)
    >>> state, timestep = env.step(state, jnp.int32(0))  # bet
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
from envs.myspaces import Discrete, Box

@chex.dataclass
class EnvState:
    """Environment state for Kuhn poker game.
    
    Attributes:
        key: PRNG key for random number generation
        current_player: ID of current player (0 or 1)
        done: Whether the game has ended
        step_cnt: Number of steps taken in the game
        cards: Cards for each player (0=J, 1=Q, 2=K)
        bets: Bets placed by each player (0 or 1)
        phase: Game phase (0=initial, 1=facing bet, 2=after pass, 3=facing bet after pass)
    """
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    cards: chex.Array
    bets: chex.Array
    phase: chex.Numeric


class KuhnPoker(env_types.BaseEnv):
    """Kuhn poker environment for two players.
    
    A simplified poker game with three cards (J, Q, K) and limited betting.
    Players are dealt one card each and take turns betting or passing.
    The environment uses JAX for efficient computation.
    
    Game Rules:
    -----------
    Setup:
    - 3 cards: Jack (J=0), Queen (Q=1), King (K=2)
    - 2 players, each dealt 1 card (no replacement)
    - One card remains unused
    - Each player antes 1 unit before the game starts
    
    Gameplay:
    1. Player 1 acts first (chosen randomly)
    2. Each player can either:
       - Bet: Place an additional 1 unit bet
       - Pass/Check: Take no action (or fold if facing a bet)
    
    Betting Sequences:
    - Pass-Pass: Game ends, highest card wins 1 unit
    - Bet-Pass: Bettor wins 1 unit (opponent folds)
    - Pass-Bet-Pass: Bettor wins 1 unit (opponent folds)
    - Bet-Bet or Pass-Bet-Bet: Game ends, highest card wins 2 units
    
    Winning:
    - If both players have equal bets: Player with higher card wins
    - If bets are unequal: Player who bet more wins (opponent folded)
    - Winner receives positive reward, loser receives negative reward
    - Reward magnitude equals the pot size (1 or 2 units)
    
    Information:
    - Players know their own card but not opponent's card
    - Perfect recall: Players know the complete action sequence history
    - Observations encode the exact action sequence state from current player's POV
    """
    
    @property
    def env_name(self) -> str:
        """Environment name identifier."""
        return "kuhn_poker"

    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return 2

    @property
    def action_space(self) -> Discrete:
        """Action space for Kuhn poker.
        
        The action space consists of 2 discrete actions:
        - 0: Bet (place a bet of 1 unit)
        - 1: Pass/Check (no bet, or fold if facing a bet)
        
        Returns:
            Discrete: Action space with 2 categories
            
        Note:
            The same action (pass) can mean different things depending on context:
            - Check: Pass when no bet has been made
            - Fold: Pass when facing a bet (forfeit the hand)
        """
        return Discrete(num_categories=2)

    @property
    def observation_space(self) -> Box:
        """Observation space for Kuhn poker with perfect recall.
        
        The observation is a 7-dimensional binary vector with one-hot encodings:
        - [0:3]: Player's card (J=0, Q=1, K=2) - one-hot encoded
        - [3:7]: Action sequence state from current player's POV - one-hot encoded
          - State 0: [] (game start, I act first)
          - State 1: [P] (opponent passed, I act second)
          - State 2: [B] (opponent bet, I act second)  
          - State 3: [P,B] (I passed, opponent bet, I act again)
        
        Returns:
            Box: 7-dimensional observation space with binary values
            
        Example:
            [0, 1, 0, 0, 0, 1, 0] means:
            - Player has Queen (position 1)
            - Game state is [P] - opponent passed, I act second (position 5)
        """
        return Box(low=0, high=1, shape=(7,), dtype=jnp.int32)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, env_types.TimeStep]:
        """Reset the environment to initial state.
        
        Args:
            key: PRNG key for random initialization
            
        Returns:
            Tuple of (initial_state, initial_timestep)
        """
        key, player_key, deal_key = jax.random.split(key, 3)
        starting_player = jax.random.bernoulli(player_key).astype(jnp.int32)
        cards = jax.random.choice(deal_key, jnp.arange(3), shape=(2,), replace=False)

        initial_state = EnvState(
            key=key,
            current_player=starting_player,
            done=jnp.bool_(False),
            step_cnt=jnp.int32(0),
            cards=cards,
            bets=jnp.zeros((2,), dtype=jnp.int32),
            phase=jnp.int32(0)
        )

        initial_timestep = env_types.TimeStep(
            reward=jnp.zeros((2,), dtype=jnp.float32),
            done=initial_state.done,
            observation=self._get_obs(initial_state),
            action_mask=jnp.ones((2,), dtype=bool),
            current_player=initial_state.current_player,
            info={"step_cnt": initial_state.step_cnt}
        )

        return initial_state, initial_timestep
    
    @partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action: env_types.Action) -> Tuple[EnvState, env_types.TimeStep]:
        """Execute one step of the environment.
        
        Args:
            state: Current environment state
            action: Action to take (0=bet, 1=pass)
            
        Returns:
            Tuple of (new_state, timestep)
        """
        chex.assert_shape(action.shape, ())

        updated_state = jax.lax.cond(
            state.done,
            lambda s: s,
            lambda s: jax.lax.switch(
                s.phase,
                [
                    lambda ss: self._step_phase0(ss, action),
                    lambda ss: self._step_phase1(ss, action),
                    lambda ss: self._step_phase2(ss, action),
                    lambda ss: self._step_phase1(ss, action),  # Phase 3 same as 1
                ],
                s
            ),
            state
        )

        rewards = self._compute_rewards(updated_state)

        action_mask = jnp.logical_not(updated_state.done) * jnp.ones((2,), dtype=bool)

        final_timestep = env_types.TimeStep(
            reward=rewards,
            done=updated_state.done,
            observation=self._get_obs(updated_state),
            action_mask=action_mask,
            current_player=updated_state.current_player,
            info={"step_cnt": updated_state.step_cnt}
        )
        return updated_state, final_timestep
    
    def _step_phase0(self, state: EnvState, action: chex.Numeric) -> EnvState:
        """Handle actions in initial phase."""
        my_idx = state.current_player
        new_bets = jax.lax.cond(
            action == 0,
            lambda b: b.at[my_idx].set(1),
            lambda b: b,
            state.bets
        )
        new_phase = jnp.where(action == 0, 1, 2)
        new_player = 1 - state.current_player
        return state.replace(
            bets=new_bets,
            current_player=new_player,
            phase=new_phase,
            step_cnt=state.step_cnt + 1
        )
    
    def _step_phase1(self, state: EnvState, action: chex.Numeric) -> EnvState:
        """Handle actions when facing a bet (phases 1 and 3)."""
        my_idx = state.current_player
        new_bets = jax.lax.cond(
            action == 0,
            lambda b: b.at[my_idx].set(1),
            lambda b: b,
            state.bets
        )
        return state.replace(
            bets=new_bets,
            done=jnp.bool_(True),
            step_cnt=state.step_cnt + 1
        )
    
    def _step_phase2(self, state: EnvState, action: chex.Numeric) -> EnvState:
        """Handle actions after initial pass."""
        my_idx = state.current_player
        new_bets = jax.lax.cond(
            action == 0,
            lambda b: b.at[my_idx].set(1),
            lambda b: b,
            state.bets
        )
        new_phase = jnp.where(action == 0, 3, state.phase)
        new_player = jnp.where(action == 0, 1 - state.current_player, state.current_player)
        new_done = jnp.where(action == 0, jnp.bool_(False), jnp.bool_(True))
        return state.replace(
            bets=new_bets,
            current_player=new_player,
            phase=new_phase,
            done=new_done,
            step_cnt=state.step_cnt + 1
        )
    
    def _compute_rewards(self, state: EnvState) -> chex.Array:
        """Compute rewards based on state."""
        return jax.lax.cond(
            state.done,
            self._get_terminal_rewards,
            lambda _: jnp.zeros((2,), dtype=jnp.float32),
            state
        )
    
    def _get_terminal_rewards(self, state: EnvState) -> chex.Array:
        """Calculate terminal rewards."""
        bet0, bet1 = state.bets
        equal_bets = (bet0 == bet1)
        mag = jax.lax.cond(equal_bets, lambda: 1 + bet0, lambda: 1)
        winner = jax.lax.cond(
            equal_bets,
            lambda: jnp.argmax(state.cards).astype(jnp.int32),
            lambda: jnp.argmax(state.bets).astype(jnp.int32)
        )
        rewards = jnp.zeros((2,), dtype=jnp.float32)
        rewards = rewards.at[winner].set(mag)
        loser = 1 - winner
        rewards = rewards.at[loser].set(-mag)
        return rewards
    
    def _get_obs(self, state: EnvState) -> chex.Array:
        """Get observation for current player with perfect recall.
        
        Encodes the player's card and the action sequence state from their POV.
        Action sequence states:
        - 0: [] (game start, I act first)
        - 1: [P] (opponent passed, I act second)
        - 2: [B] (opponent bet, I act second)  
        - 3: [P,B] (I passed, opponent bet, I act again)
        """
        player = state.current_player
        card = state.cards[player]
        
        # Map phase to action sequence state from current player's POV
        phase_to_sequence_state = jnp.array([0, 2, 1, 3])
        sequence_state = phase_to_sequence_state[state.phase]
        
        obs = jnp.zeros((7,), dtype=jnp.int32)
        obs = obs.at[card].set(1)  # Card encoding [0:3]
        obs = obs.at[3 + sequence_state].set(1)  # Sequence state encoding [3:7]
        return obs

