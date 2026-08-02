"""Tic-tac-toe environment implementation using JAX.

This module provides a JAX-based implementation of the classic tic-tac-toe game
for multi-agent reinforcement learning. The environment follows the BaseEnv
interface and supports two players competing on a 3x3 grid.

Example:
    >>> env = TicTacToe()
    >>> key = jax.random.key(42)
    >>> state, timestep = env.reset(key)
    >>> state, timestep = env.step(state, jnp.int32(4))  # center square
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
    """Environment state for tic-tac-toe game.
    
    Attributes:
        key: PRNG key for random number generation
        current_player: ID of current player (0 or 1)
        done: Whether the game has ended
        step_cnt: Number of steps taken in the game
        board: 3x3 game board (-1=empty, 0=player0, 1=player1)
        winner: ID of winner (-1=no winner, 0=player0, 1=player1)
    """
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    board: chex.Array
    winner: chex.Numeric


class TicTacToe(env_types.BaseEnv):
    """Tic-tac-toe environment for two players.
    
    A classic 3x3 tic-tac-toe game where two players take turns placing
    their marks. The first player to get three marks in a row (horizontally,
    vertically, or diagonally) wins. Invalid moves result in immediate loss.
    
    The environment uses JAX for efficient computation and supports
    vectorized operations for game logic.
    """
    
    @property
    def env_name(self) -> str:
        """Environment name identifier."""
        return "tic_tac_toe"

    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return 2

    @property
    def action_space(self) -> Discrete:
        """Action space: 9 discrete actions for 3x3 board positions."""
        return Discrete(num_categories=9)

    @property
    def observation_space(self) -> Box:
        """Observation space: 3x3 board with values -1 (empty), 0, or 1."""
        return Box(low=-1, high=1, shape=(3, 3), dtype=jnp.int32)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, env_types.TimeStep]:
        """Reset the environment to initial state.
        
        Args:
            key: PRNG key for random initialization
            
        Returns:
            Tuple of (initial_state, initial_timestep)
        """
        key, player_key = jax.random.split(key)
        starting_player = jax.random.bernoulli(player_key).astype(jnp.int32)

        initial_state = EnvState(
            key=key,
            current_player=starting_player,
            done=jnp.bool(False),
            step_cnt=jnp.int32(0),
            board=-jnp.ones((3, 3), dtype=jnp.int32),
            winner=jnp.int32(-1)
        )

        initial_timestep = env_types.TimeStep(
            reward=jnp.zeros((2,), dtype=jnp.float32),
            done=initial_state.done,
            observation=self._get_observation_for_player(initial_state.board, initial_state.current_player),
            action_mask=(initial_state.board == -1).flatten(),
            current_player=initial_state.current_player,
            info={"step_cnt": initial_state.step_cnt}
        )

        return initial_state, initial_timestep
    
    @partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action: env_types.Action) -> Tuple[EnvState, env_types.TimeStep]:
        """Execute one step of the environment.
        
        Args:
            state: Current environment state
            action: Action to take (0-8, representing board positions)
            
        Returns:
            Tuple of (new_state, timestep)
        """
        chex.assert_shape(action.shape, ())
        row, col = action // 3, action % 3

        # Check if action is valid
        is_valid_action = (action >= 0) & (action < 9) & (state.board[row, col] == -1)

        # Apply action using conditional logic
        updated_state = jax.lax.cond(
            ~state.done,  # Game is still ongoing
            lambda s: jax.lax.cond(
                is_valid_action,  # Action is valid
                partial(self._apply_action, row, col),
                self._handle_invalid_action,
                s
            ),
            lambda s: s,  # Don't modify state when game is done
            state
        )

        # Calculate rewards based on game outcome
        reward_lookup = jnp.array([0, 1, -1])  # [draw, player0_win, player1_win]
        player0_reward = reward_lookup[updated_state.winner + 1]
        player1_reward = -reward_lookup[updated_state.winner + 1]
        rewards = jnp.array([player0_reward, player1_reward], dtype=jnp.float32)

        final_timestep = env_types.TimeStep(
            reward=rewards,
            done=updated_state.done,
            observation=self._get_observation_for_player(updated_state.board, updated_state.current_player),
            action_mask=(updated_state.board == -1).flatten(),
            current_player=updated_state.current_player,
            info={"step_cnt": updated_state.step_cnt}
        )
        return updated_state, final_timestep
    
    def _handle_invalid_action(self, state: EnvState) -> EnvState:
        """Handle invalid action by ending game with opponent as winner.
        
        Args:
            state: Current environment state
            
        Returns:
            Updated state with opponent as winner
        """
        return state.replace(
            winner=1 - state.current_player,
            done=jnp.bool(True),
            step_cnt=state.step_cnt + 1,
            current_player=1 - state.current_player,
        )

    def _apply_action(self, row: chex.Numeric, col: chex.Numeric, state: EnvState) -> EnvState:
        """Apply valid action to the game state.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2) 
            state: Current environment state
            
        Returns:
            Updated state after applying the action
        """
        # Update board with immutable operation
        updated_board = state.board.at[row, col].set(state.current_player)

        # Check win condition
        has_won = self._player_win(updated_board, state.current_player)

        # Check for draw when board is full
        is_draw = jnp.all(updated_board != -1)

        # Return updated state using immutable replace
        return state.replace(
            board=updated_board,
            winner=jnp.where(has_won, state.current_player, -1),
            done=has_won | is_draw,
            step_cnt=state.step_cnt + 1,
            current_player=1 - state.current_player
        )
    
    def _get_observation_for_player(self, board: chex.Array, player: chex.Numeric) -> chex.Array:
        """Get board observation from the perspective of the given player.
        
        Each player sees themselves as 0 and opponent as 1 in the observation.
        
        Args:
            board: 3x3 game board (-1=empty, 0=player0, 1=player1)
            player: Player ID (0 or 1) requesting the observation
            
        Returns:
            Board from player's perspective (-1=empty, 0=self, 1=opponent)
        """
        # If player is 0, keep board as-is
        # If player is 1, swap 0 and 1 values
        return jnp.where(
            player == 0,
            board,  # Player 0 sees original board
            jnp.where(board == -1, -1,  # Keep empty cells as -1
                     jnp.where(board == 0, 1,  # Opponent (player 0) becomes 1
                              0))  # Self (player 1) becomes 0
        )
    
    def _player_win(self, board: chex.Array, player: chex.Numeric) -> chex.Numeric:
        """Check if the player has won on the given board.
        
        Args:
            board: 3x3 game board
            player: Player ID to check for win
            
        Returns:
            Boolean indicating if player has won
        """
        # Check horizontal wins using vectorized operations
        horizontal_wins = jnp.all(board == player, axis=1).any()
        
        # Check vertical wins using vectorized operations
        vertical_wins = jnp.all(board == player, axis=0).any()
        
        # Check diagonal wins
        main_diagonal = jnp.all(jnp.diag(board) == player)
        anti_diagonal = jnp.all(jnp.diag(jnp.fliplr(board)) == player)
        diagonal_wins = main_diagonal | anti_diagonal
        
        return horizontal_wins | vertical_wins | diagonal_wins


if __name__ == "__main__":
    env = TicTacToe()
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    state, timestep = env.reset(subkey)
    print(f"{state.board}")
    action = jnp.int32(0)

    while not state.done:
        key, subkey = jax.random.split(key)
        state, timestep = env.step(state, action)
        print(f"action: {action}")
        print(f"{state.board}")
        action += 1

    print(state)