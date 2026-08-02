"""Liar's Dice environment implementation using JAX.

Single-hand variant with 5 dice and 6 sides per die.
Reference: https://en.wikipedia.org/wiki/Liar%27s_dice

Example:
    >>> env = LiarsDice()
    >>> key = jax.random.key(42)
    >>> state, timestep = env.reset(key)
    >>> state, timestep = env.step(state, jnp.int32(1))  # raise bid
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
from envs.myspaces import Discrete, Box, Dict

MAX_HISTORY_LENGTH = 32

@chex.dataclass
class EnvState:
    """Environment state for Liar's Dice game.
    
    Attributes:
        key: PRNG key for random number generation
        current_player: ID of current player (0 or 1)
        done: Whether the game has ended
        step_cnt: Number of steps taken in the game
        dice: Dice values for each player (2, 5) - values 1-6
        bid_history_player: History of which player made each bid (MAX_HISTORY_LENGTH,)
        bid_history_quantity: History of bid quantities (MAX_HISTORY_LENGTH,)
        bid_history_face: History of bid faces (MAX_HISTORY_LENGTH,)
        current_bid_quantity: Current bid quantity (1-10)
        current_bid_face: Current bid face (1-6)
        winner: Winner of the game (-1 if ongoing, 0 or 1 if finished)
    """
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    dice: chex.Array
    bid_history_player: chex.Array
    bid_history_quantity: chex.Array
    bid_history_face: chex.Array
    current_bid_quantity: chex.Numeric
    current_bid_face: chex.Numeric
    winner: chex.Numeric


class LiarDice(env_types.BaseEnv):
    """Liar's Dice environment for two players.
    
    A dice bidding game where players bid on the total number of dice faces
    across all players' dice. Players can raise the bid or challenge the
    previous bid.
    
    Game Rules:
    -----------
    Setup:
    - 2 players, each with 5 six-sided dice (values 1-6)
    - All dice are rolled secretly at the start
    - Starting player chosen randomly
    
    Gameplay:
    1. Players take turns bidding on total number of a specific face across all dice
    2. Each bid must be higher than the previous (either higher quantity or same/higher quantity with higher face)
    3. A player can either:
       - Raise: Make a higher bid
       - Challenge: Call "Liar!" on the previous bid
    
    Bidding Rules:
    - First bid can be any quantity (1-10) and face (1-6)
    - Subsequent bids must be strictly higher:
      * Higher quantity with same or higher face
      * Same quantity with higher face
    
    Challenge Resolution:
    - All dice are revealed
    - Count total occurrences of the challenged face across all dice
    - If actual count >= bid quantity: challenger loses
    - If actual count < bid quantity: bidder loses
    
    Actions:
    - Action 0: Challenge previous bid
    - Actions 1-60: Raise bid (encoded as (quantity-1)*6 + (face-1) + 1)
      * quantity: 1-10, face: 1-6
      * Example: bid "3 fours" = (3-1)*6 + (4-1) + 1 = 16
    
    Observations:
    - Perfect recall: Full history of all bids made
    - Player's own dice face counts
    - Current game state
    """
    
    @property
    def env_name(self) -> str:
        """Environment name identifier."""
        return "liars_dice"

    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return 2

    @property
    def action_space(self) -> Discrete:
        """Action space for Liar's Dice.
        
        The action space consists of 61 discrete actions:
        - Action 0: Challenge the previous bid
        - Actions 1-60: Raise the bid
          * Encoded as (quantity-1)*6 + (face-1) + 1
          * quantity: 1-10, face: 1-6
        
        Returns:
            Discrete: Action space with 61 categories
        """
        return Discrete(num_categories=61)

    @property
    def observation_space(self) -> Dict:
        """Observation space for Liar's Dice.
        
        The observation is a dictionary containing:
        - 'bid_history_player': (MAX_HISTORY_LENGTH,) array of which player made each bid (0=self, 1=opponent, -1=padding)
        - 'bid_history_quantity': (MAX_HISTORY_LENGTH,) array of bid quantities in history (-1=padding)
        - 'bid_history_face': (MAX_HISTORY_LENGTH,) array of bid faces in history (-1=padding)
        - 'own_dice_counts': (6,) array of count of each face (1-6) for current player
        
        Note: All observations are from current player's perspective (self=0, opponent=1)
        
        Returns:
            Dict: Dictionary observation space with perfect recall
        """
        return Dict({
            'bid_history_player': Box(low=-1, high=1, shape=(MAX_HISTORY_LENGTH,), dtype=jnp.int32),
            'bid_history_quantity': Box(low=-1, high=10, shape=(MAX_HISTORY_LENGTH,), dtype=jnp.int32),
            'bid_history_face': Box(low=-1, high=6, shape=(MAX_HISTORY_LENGTH,), dtype=jnp.int32),
            'own_dice_counts': Box(low=0, high=5, shape=(6,), dtype=jnp.int32)
        })

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, env_types.TimeStep]:
        """Reset the environment to initial state.
        
        Args:
            key: PRNG key for random initialization
            
        Returns:
            Tuple of (initial_state, initial_timestep)
        """
        key, player_key, dice_key = jax.random.split(key, 3)
        
        # Choose starting player randomly
        starting_player = jax.random.bernoulli(player_key).astype(jnp.int32)
        
        # Roll dice for both players (values 1-6)
        dice = jax.random.randint(dice_key, (2, 5), minval=1, maxval=7, dtype=jnp.int32)

        initial_state = EnvState(
            key=key,
            current_player=starting_player,
            done=jnp.bool_(False),
            step_cnt=jnp.int32(0),
            dice=dice,
            bid_history_player=jnp.full((MAX_HISTORY_LENGTH,), -1, dtype=jnp.int32),
            bid_history_quantity=jnp.full((MAX_HISTORY_LENGTH,), -1, dtype=jnp.int32),
            bid_history_face=jnp.full((MAX_HISTORY_LENGTH,), -1, dtype=jnp.int32),
            current_bid_quantity=jnp.int32(0),
            current_bid_face=jnp.int32(0),
            winner=jnp.int32(-1)
        )

        initial_timestep = env_types.TimeStep(
            reward=jnp.zeros((2,), dtype=jnp.float32),
            done=initial_state.done,
            observation=self._get_obs(initial_state),
            action_mask=self._get_action_mask(initial_state),
            current_player=initial_state.current_player,
            info={"step_cnt": initial_state.step_cnt}
        )

        return initial_state, initial_timestep
    
    @partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action: env_types.Action) -> Tuple[EnvState, env_types.TimeStep]:
        """Execute one step of the environment.
        
        Args:
            state: Current environment state
            action: Action to take (0=challenge, 1-60=raise bid)
            
        Returns:
            Tuple of (new_state, timestep)
        """
        chex.assert_shape(action.shape, ())

        # If game is done, return current state
        def stay_done(s):
            return s

        # Process action if game is not done
        def process_action(s):
            return jax.lax.cond(
                action == 0,
                lambda state: self._process_challenge(state),
                lambda state: self._process_raise(state, action),
                s
            )

        updated_state = jax.lax.cond(
            state.done,
            stay_done,
            process_action,
            state
        )

        rewards = self._compute_rewards(updated_state)
        action_mask = self._get_action_mask(updated_state)

        final_timestep = env_types.TimeStep(
            reward=rewards,
            done=updated_state.done,
            observation=self._get_obs(updated_state),
            action_mask=action_mask,
            current_player=updated_state.current_player,
            info={"step_cnt": updated_state.step_cnt}
        )
        
        return updated_state, final_timestep
    
    def _process_challenge(self, state: EnvState) -> EnvState:
        """Process a challenge action."""
        # Count actual occurrences of the current bid face
        all_dice = state.dice.flatten()
        actual_count = jnp.sum(all_dice == state.current_bid_face)
        
        # Determine winner: if actual_count >= bid_quantity, challenger loses
        challenger_loses = actual_count >= state.current_bid_quantity
        winner = jnp.where(
            challenger_loses,
            1 - state.current_player,  # Previous bidder wins
            state.current_player       # Challenger wins
        )
        
        return state.replace(
            done=jnp.bool_(True),
            step_cnt=state.step_cnt + 1,
            winner=winner
        )
    
    def _process_raise(self, state: EnvState, action: chex.Numeric) -> EnvState:
        """Process a raise action."""
        # Decode action to quantity and face
        action_idx = action - 1  # Convert to 0-based index
        quantity = (action_idx // 6) + 1  # 1-10
        face = (action_idx % 6) + 1       # 1-6
        
        # Update bid history using circular buffer
        buffer_idx = state.step_cnt % MAX_HISTORY_LENGTH
        new_bid_history_player = state.bid_history_player.at[buffer_idx].set(state.current_player)
        new_bid_history_quantity = state.bid_history_quantity.at[buffer_idx].set(quantity)
        new_bid_history_face = state.bid_history_face.at[buffer_idx].set(face)
        
        # Switch to next player
        next_player = 1 - state.current_player
        
        return state.replace(
            current_player=next_player,
            step_cnt=state.step_cnt + 1,
            bid_history_player=new_bid_history_player,
            bid_history_quantity=new_bid_history_quantity,
            bid_history_face=new_bid_history_face,
            current_bid_quantity=quantity,
            current_bid_face=face
        )
    
    def _get_action_mask(self, state: EnvState) -> chex.Array:
        """Get valid action mask for current state."""
        
        def done_mask(s):
            return jnp.zeros((61,), dtype=bool)
        
        def active_mask(s):
            def first_move_mask(s):
                mask = jnp.ones((61,), dtype=bool)
                return mask.at[0].set(False)  # Can't challenge on first move
            
            def normal_mask(s):
                # Can always challenge (except first move)
                mask = jnp.zeros((61,), dtype=bool)
                mask = mask.at[0].set(True)  # Challenge is always valid
                
                # Check which raises are valid (must be higher than current bid)
                current_bid_val = (s.current_bid_quantity - 1) * 6 + (s.current_bid_face - 1)
                
                # Vectorized computation for all raise actions (1-60)
                action_indices = jnp.arange(60)  # 0 to 59 for actions 1-60
                quantities = (action_indices // 6) + 1  # 1-10
                faces = (action_indices % 6) + 1  # 1-6
                bid_vals = (quantities - 1) * 6 + (faces - 1)
                
                # Valid if strictly higher than current bid
                valid_raises = bid_vals > current_bid_val
                mask = mask.at[1:61].set(valid_raises)
                
                return mask
            
            return jax.lax.cond(
                s.step_cnt == 0,
                first_move_mask,
                normal_mask,
                s
            )
        
        return jax.lax.cond(
            state.done,
            done_mask,
            active_mask,
            state
        )
    
    def _compute_rewards(self, state: EnvState) -> chex.Array:
        """Compute rewards based on game outcome."""
        return jax.lax.cond(
            state.done,
            lambda s: self._get_terminal_rewards(s),
            lambda s: jnp.zeros((2,), dtype=jnp.float32),
            state
        )
    
    def _get_terminal_rewards(self, state: EnvState) -> chex.Array:
        """Calculate terminal rewards."""
        rewards = jnp.zeros((2,), dtype=jnp.float32)
        rewards = rewards.at[state.winner].set(1.0)
        loser = 1 - state.winner
        rewards = rewards.at[loser].set(-1.0)
        return rewards
    
    def _get_obs(self, state: EnvState) -> dict:
        """Get observation for current player.
        
        Observations are from current player's perspective:
        - Current player is always seen as player 0
        - Opponent is always seen as player 1
        - Padding entries remain as -1
        """
        player = state.current_player
        
        # Get own dice face counts (1-6) - vectorized
        faces = jnp.arange(1, 7)  # [1, 2, 3, 4, 5, 6]
        own_dice_counts = jax.vmap(lambda face: jnp.sum(state.dice[player] == face))(faces)
        
        # Transform bid history to current player's perspective
        # Current player becomes 0, opponent becomes 1, padding stays -1
        transformed_bid_history_player = jnp.where(
            state.bid_history_player == -1,  # Padding
            -1,
            jnp.where(
                state.bid_history_player == player,  # Current player -> 0
                0,
                1  # Opponent -> 1
            )
        )
        
        obs = {
            'bid_history_player': transformed_bid_history_player,
            'bid_history_quantity': state.bid_history_quantity,
            'bid_history_face': state.bid_history_face,
            'own_dice_counts': own_dice_counts
        }
        
        return obs
