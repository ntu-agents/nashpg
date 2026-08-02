"""Leduc poker environment implementation using JAX with perfect recall.

This module provides a JAX-based implementation of Leduc poker
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
    """Environment state for Leduc poker game.
    
    Attributes:
        key: PRNG key for random number generation
        current_player: ID of current player (0 or 1)
        done: Whether the game has ended
        step_cnt: Number of steps taken in the game
        private_cards: Private cards for each player [player0_card, player1_card] (0-5)
        board_card: Community board card (0-5, or -1 if not revealed)
        current_round: Current betting round (0 for round 1, 1 for round 2)
        starting_player: Which player goes first (0 or 1)
        pot: Current pot size for each player [player0_contribution, player1_contribution]
        betting_history: History of actions taken [round, position, action] - flattened array
        betting_position: Current position within the betting round
        last_bet_size: Size of the last bet/raise
        terminal_reason: Reason for termination (0=ongoing, 1=fold, 2=showdown)
    """
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    private_cards: chex.Array
    board_card: chex.Numeric
    current_round: chex.Numeric
    starting_player: chex.Numeric
    pot: chex.Array
    betting_history: chex.Array  # Shape (2, 4) - 2 rounds, max 4 actions per round
    betting_position: chex.Numeric
    last_bet_size: chex.Numeric
    terminal_reason: chex.Numeric



class LeducPoker(env_types.BaseEnv):
    """Leduc poker environment for two players.
    
    A simplified poker game with two suits of three cards each and two betting rounds.
    Players are dealt one private card each, then a community board card is revealed.
    The environment uses JAX for efficient computation.
    
    Game Rules:
    -----------
    Setup:
    - 6 cards: Two suits (e.g., Hearts and Spades) with three ranks each (J=0, Q=1, K=2)
    - 2 players, each dealt 1 private card (no replacement)
    - One community board card revealed in round 2
    - Both players start with 1 unit already in the pot
    
    Gameplay:
    1. Round 1 (Private cards only):
       - Each player knows only their own card
       - Player 1 acts first (chosen randomly)
       - Raise amount: 2 units
       - Two-bet maximum per round
    
    2. Round 2 (Board card revealed):
       - One community card is dealt face-up
       - Players can now make pairs with their private card
       - Raise amount: 4 units
       - Two-bet maximum per round
    
    Actions per round:
    - Call/Check: Match current bet or take no action if no bet
    - Raise: Increase bet by fixed amount (2 in round 1, 4 in round 2)
    - Fold: Forfeit hand and lose all money in pot
    
    Hand Rankings:
    - Pair: Private card matches board card (by rank)
    - High Card: No pair, highest card wins (King > Queen > Jack)
    - Suit does not matter for hand strength
    
    Winning:
    - At showdown: Player with better hand wins entire pot
    - By folding: Remaining player wins entire pot
    - Winner receives positive reward, loser receives negative reward
    - Reward magnitude equals the pot size
    
    Information:
    - Players know their own private card and the board card (when revealed)
    - Perfect recall: Players know the complete betting sequence history
    - Observations encode the betting state and visible cards from current player's POV
    """    
    
    @property
    def env_name(self) -> str:
        """Environment name identifier."""
        return "leduc_poker"

    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return 2

    @property
    def action_space(self) -> Discrete:
        """Action space for Leduc poker.
        
        The action space consists of 3 discrete actions:
        - 0: Fold (forfeit the hand and lose all money in pot)
        - 1: Call/Check (match current bet or take no action if no bet)
        - 2: Raise (increase bet by fixed amount: 2 in round 1, 4 in round 2)
        
        Returns:
            Discrete: Action space with 3 categories
        """
        return Discrete(num_categories=3)

    @property
    def observation_space(self) -> Box:
        """Observation space for Leduc poker with perfect recall from current player's POV.
        
        The observation is a 49-dimensional binary vector with one-hot encodings:
        - [0:6]: My private card - one-hot (J♠=0, Q♠=1, K♠=2, J♥=3, Q♥=4, K♥=5)
        - [6:13]: Board card - one-hot (6 cards + 1 for "not revealed")
        - [13:15]: Current round - one-hot (round 1=0, round 2=1)
        - [15:17]: Do I act first in any round - one-hot (I go first=0, opponent goes first=1)
        - [17:33]: Round 1 betting history from my POV - 4 positions × 4 actions each
        - [33:49]: Round 2 betting history from my POV - 4 positions × 4 actions each  
        
        Player POV betting positions:
        - Position 0: My first action in round
        - Position 1: Opponent's first action in round  
        - Position 2: My second action in round (if any)
        - Position 3: Opponent's second action in round (if any)
        
        Actions encoding: 0=Fold, 1=Call/Check, 2=Raise, 3=No action yet
        
        Returns:
            Box: 49-dimensional observation space with binary values from current player's perspective
        """
        return Box(low=0, high=1, shape=(49,), dtype=jnp.int32)

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
        
        # Deal cards: 2 private cards + 1 board card from 6-card deck
        dealt_cards = jax.random.choice(deal_key, jnp.arange(6), shape=(3,), replace=False)
        private_cards = dealt_cards[:2]
        board_card = dealt_cards[2]

        # Initialize state
        initial_state = EnvState(
            key=key,
            current_player=starting_player,
            done=jnp.bool_(False),
            step_cnt=jnp.int32(0),
            private_cards=private_cards,
            board_card=board_card,
            current_round=jnp.int32(0),  # Start in round 1 (pre-flop)
            starting_player=starting_player,
            pot=jnp.array([1, 1], dtype=jnp.int32),  # Both players ante 1 unit
            betting_history=jnp.full((2, 4), 3, dtype=jnp.int32),  # 3 = "no action yet"
            betting_position=jnp.int32(0),
            last_bet_size=jnp.int32(0),
            terminal_reason=jnp.int32(0)  # 0 = ongoing
        )
        
        # Get initial observation and action mask
        observation = self._get_obs(initial_state)
        action_mask = self._get_action_mask(initial_state)
        
        initial_timestep = env_types.TimeStep(
            reward=jnp.zeros(2, dtype=jnp.float32),
            done=initial_state.done,
            observation=observation,
            action_mask=action_mask,
            current_player=initial_state.current_player,
            info={"step_cnt": initial_state.step_cnt}
        )
        
        return initial_state, initial_timestep
    
    @partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action: env_types.Action) -> Tuple[EnvState, env_types.TimeStep]:
        """Execute one step of the environment.
        
        Args:
            state: Current environment state
            action: Action to take (0=Fold, 1=Call/Check, 2=Raise)
            
        Returns:
            Tuple of (new_state, timestep)
        """
        chex.assert_shape(action.shape, ())

        action_mask = self._get_action_mask(state)
        is_action_valid = action_mask[action]

        # If game is done, don't update state
        updated_state = jax.lax.cond(
            state.done,
            lambda s: s,
            lambda s: jax.lax.cond(
                is_action_valid,
                lambda s: self._update_state(s, action),
                lambda s: s,
                s
            ),
            state
        )

        rewards = self._compute_rewards(updated_state)
        action_mask = self._get_action_mask(updated_state)

        timestep = env_types.TimeStep(
            reward=rewards,
            done=updated_state.done,
            observation=self._get_obs(updated_state),
            action_mask=action_mask,
            current_player=updated_state.current_player,
            info={"step_cnt": updated_state.step_cnt}
        )
        return updated_state, timestep
    
    @partial(jax.jit, static_argnums=0)
    def _get_action_mask(self, state: EnvState) -> chex.Array:
        """Get valid action mask for current player.
        
        Args:
            state: Current environment state
            
        Returns:
            Binary mask indicating valid actions [fold, call, raise]
        """
        # If game is done, no actions are valid
        done_mask = jnp.array([False, False, False], dtype=jnp.bool_)
        
        # If game is not done, determine valid actions
        # Fold is only valid when facing a bet (opponent has more in pot)
        # Call is always valid when game is ongoing
        # Raise is valid if we haven't exceeded 2-bet limit per round
        current_round = state.current_round
        betting_history_round = state.betting_history[current_round]

        opponent = 1 - state.current_player
        can_fold = state.pot[opponent] > state.pot[state.current_player]

        # Count number of raises in current round
        raise_count = jnp.sum(betting_history_round == 2)
        can_raise = raise_count < 2  # Max 2 raises per round

        ongoing_mask = jnp.array([can_fold, True, can_raise], dtype=jnp.bool_)
        
        return jnp.where(state.done, done_mask, ongoing_mask)
    
    @partial(jax.jit, static_argnums=0)
    def _update_state(self, state: EnvState, action: env_types.Action) -> EnvState:
        """Update state based on action taken.
        
        Args:
            state: Current environment state
            action: Action taken (0=Fold, 1=Call/Check, 2=Raise)
            
        Returns:
            Updated environment state
        """
        current_player = state.current_player
        current_round = state.current_round
        betting_position = state.betting_position
        
        # Update betting history
        new_betting_history = state.betting_history.at[current_round, betting_position].set(action)
        
        # Handle different actions
        new_pot = state.pot
        new_last_bet_size = state.last_bet_size
        new_done = state.done
        new_terminal_reason = state.terminal_reason
        
        # Action 0: Fold
        def handle_fold():
            return (
                state.pot,  # pot stays same
                state.last_bet_size,  # bet size stays same
                jnp.bool_(True),  # game ends
                jnp.int32(1)  # terminal reason = fold
            )
        
        # Action 1: Call/Check
        def handle_call():
            # Calculate call amount needed
            opponent = 1 - current_player
            call_amount = state.pot[opponent] - state.pot[current_player]
            new_pot_call = state.pot.at[current_player].add(call_amount)
            
            return (
                new_pot_call,
                state.last_bet_size,
                state.done,
                state.terminal_reason
            )
        
        # Action 2: Raise
        def handle_raise():
            # Raise amounts: 2 in round 1, 4 in round 2
            raise_amount = jnp.where(current_round == 0, 2, 4)
            
            # First call to match, then raise
            opponent = 1 - current_player
            call_amount = state.pot[opponent] - state.pot[current_player]
            total_add = call_amount + raise_amount
            new_pot_raise = state.pot.at[current_player].add(total_add)
            
            return (
                new_pot_raise,
                jnp.int32(raise_amount),
                state.done,
                state.terminal_reason
            )
        
        # Apply action
        new_pot, new_last_bet_size, new_done, new_terminal_reason = jax.lax.switch(
            action,
            [handle_fold, handle_call, handle_raise]
        )
        
        # Handle immediate game ending (fold)
        def handle_fold_end():
            return (
                current_round,  # stay in current round
                current_player,  # keep current player (who folded)
                betting_position,  # keep position
                new_done,  # game is done
                new_terminal_reason  # preserve fold reason
            )
        
        # Check if betting round is complete (natural completion)
        both_acted = betting_position >= 1
        pots_equal = new_pot[0] == new_pot[1]
        natural_round_complete = both_acted & pots_equal
        
        # Advance to next round or end game (natural progression)
        def advance_round():
            next_round = current_round + 1
            game_ends = next_round >= 2  # Only 2 rounds in Leduc
            
            return (
                jnp.where(game_ends, current_round, next_round),  # current_round
                state.starting_player,  # reset to starting player
                jnp.int32(0),  # reset betting position
                jnp.where(game_ends, jnp.bool_(True), new_done),  # done
                jnp.where(game_ends, jnp.int32(2), new_terminal_reason)  # showdown if natural end
            )
        
        def continue_round():
            return (
                current_round,
                1 - current_player,  # switch to other player
                betting_position + 1,  # advance betting position
                new_done,
                new_terminal_reason
            )
        
        # If someone folded, end immediately; otherwise check round progression
        final_round, next_player, next_position, final_done, final_terminal_reason = jax.lax.cond(
            new_done & (new_terminal_reason == 1),  # fold
            handle_fold_end,
            lambda: jax.lax.cond(natural_round_complete, advance_round, continue_round)
        )
        
        # If game ends due to fold, keep current player (the one who folded)
        final_player = jnp.where(
            final_done & (final_terminal_reason == 1),  # fold
            current_player,  # keep folding player
            next_player  # otherwise use normal switching
        )
        
        return state.replace(
            current_player=final_player,
            done=final_done,
            step_cnt=state.step_cnt + 1,
            current_round=final_round,
            pot=new_pot,
            betting_history=new_betting_history,
            betting_position=next_position,
            last_bet_size=new_last_bet_size,
            terminal_reason=final_terminal_reason
        )
    
    @partial(jax.jit, static_argnums=0)
    def _get_obs(self, state: EnvState) -> chex.Array:
        """Get observation from current player's POV.
        
        Args:
            state: Current environment state
            
        Returns:
            49-dimensional observation vector for current player
        """
        current_player = state.current_player
        obs = jnp.zeros(49, dtype=jnp.int32)
        
        # [0:6] My private card (one-hot)
        my_card = state.private_cards[current_player]
        obs = obs.at[my_card].set(1)
        
        # [6:13] Board card (one-hot, 7th position for "not revealed")
        board_revealed = state.current_round >= 1
        board_pos = jnp.where(board_revealed, state.board_card + 6, 12)  # 12 = "not revealed"
        obs = obs.at[board_pos].set(1)
        
        # [13:15] Current round (one-hot)
        obs = obs.at[13 + state.current_round].set(1)
        
        # [15:17] Do I act first (one-hot)
        i_go_first = current_player == state.starting_player
        first_pos = jnp.where(i_go_first, 15, 16)
        obs = obs.at[first_pos].set(1)
        
        # [17:33] Round 1 betting history from my POV (4 positions × 4 actions)
        # [33:49] Round 2 betting history from my POV (4 positions × 4 actions)
        # Positions: 0=my 1st, 1=opp 1st, 2=my 2nd, 3=opp 2nd
        for round_idx in range(2):
            base_idx = 17 + round_idx * 16  # 16 = 4 positions × 4 actions
            round_history = state.betting_history[round_idx]
            
            for pos in range(4):
                action = round_history[pos]
                # Determine whose action this is based on starting player and position
                acting_player = (state.starting_player + pos) % 2
                
                # Map to POV positions: even=mine, odd=opponent's
                is_my_action = acting_player == current_player
                pov_pos = jnp.where(is_my_action, pos // 2 * 2, pos // 2 * 2 + 1)
                
                # Set one-hot for the action
                action_idx = base_idx + pov_pos * 4 + action
                obs = obs.at[action_idx].set(1)
        
        return obs
    
    @partial(jax.jit, static_argnums=0)
    def _compute_rewards(self, state: EnvState) -> chex.Array:
        """Compute rewards for both players.
        
        Args:
            state: Current environment state
            
        Returns:
            Rewards for both players [player0_reward, player1_reward]
        """
        # No rewards if game is not done
        ongoing_rewards = jnp.zeros(2, dtype=jnp.float32)
        
        # If game is done, compute final rewards
        def compute_final_rewards():
            # If someone folded, opponent wins the pot
            def fold_rewards():
                folding_player = state.current_player  # current player just folded
                winner = 1 - state.current_player  # opponent wins
                
                pot_size = jnp.sum(state.pot)
                rewards = jnp.zeros(2, dtype=jnp.float32)
                rewards = rewards.at[winner].set(pot_size - state.pot[winner])
                rewards = rewards.at[folding_player].set(-state.pot[folding_player])
                return rewards
            
            # If showdown, best hand wins
            def showdown_rewards():
                hand_strength_0 = self._get_hand_strength(state.private_cards[0], state.board_card)
                hand_strength_1 = self._get_hand_strength(state.private_cards[1], state.board_card)
                
                # Winner has higher hand strength
                player_0_wins = hand_strength_0 > hand_strength_1
                player_1_wins = hand_strength_1 > hand_strength_0
                
                pot_size = jnp.sum(state.pot)
                
                # Player 0 wins
                def p0_wins():
                    rewards = jnp.zeros(2, dtype=jnp.float32)
                    rewards = rewards.at[0].set(pot_size - state.pot[0])
                    rewards = rewards.at[1].set(-state.pot[1])
                    return rewards
                
                # Player 1 wins  
                def p1_wins():
                    rewards = jnp.zeros(2, dtype=jnp.float32)
                    rewards = rewards.at[1].set(pot_size - state.pot[1])
                    rewards = rewards.at[0].set(-state.pot[0])
                    return rewards
                
                # Tie - split pot (shouldn't happen in Leduc with distinct cards)
                def tie_rewards():
                    return jnp.zeros(2, dtype=jnp.float32)  # Everyone breaks even
                
                return jax.lax.cond(
                    player_0_wins,
                    p0_wins,
                    lambda: jax.lax.cond(player_1_wins, p1_wins, tie_rewards)
                )
            
            return jax.lax.cond(
                state.terminal_reason == 1,  # fold
                fold_rewards,
                showdown_rewards
            )
        
        return jnp.where(state.done, compute_final_rewards(), ongoing_rewards)
    
    @partial(jax.jit, static_argnums=0)
    def _get_hand_strength(self, private_card: chex.Numeric, board_card: chex.Numeric) -> chex.Numeric:
        """Get hand strength for a player.
        
        Args:
            private_card: Player's private card (0-5)
            board_card: Community board card (0-5)
            
        Returns:
            Hand strength (higher is better)
        """
        # Cards are encoded as: J♠=0, Q♠=1, K♠=2, J♥=3, Q♥=4, K♥=5
        # Rank: J=0, Q=1, K=2
        private_rank = private_card % 3
        board_rank = board_card % 3
        
        # Check for pair (same rank)
        has_pair = private_rank == board_rank
        
        # Hand strength: pairs beat high cards
        # Pair strength: 100 + rank (higher rank pair beats lower rank pair)  
        # High card strength: rank (higher card beats lower card)
        pair_strength = 100 + private_rank
        high_card_strength = private_rank
        
        return jnp.where(has_pair, pair_strength, high_card_strength)