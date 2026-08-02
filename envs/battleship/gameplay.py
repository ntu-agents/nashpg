"""Play stage logic for Battleship environment."""

from typing import Tuple
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
from .state import EnvState
from .utils import TOTAL_SHIP_POSITIONS


def play_stage_act(
    state: EnvState,
    action: env_types.Action
) -> Tuple[EnvState, chex.Numeric, chex.Array]:
    """Perform action at play stage.

    Args:
        state: Current environment state
        action: Action to take (position 0-99)

    Returns:
        Tuple of (new_state, is_valid_action, step_reward)
        step_reward: shape (2,) — +1/TOTAL_SHIP_POSITIONS for hitter on a hit, else 0
    """
    play_board = state.board[state.current_player, 1]  # current player's fire board
    enemy_ship_board = state.board[1 - state.current_player, 0]  # enemy's ship board

    # Valid actions are positions not yet targeted
    action_mask = play_board == 0  # free space (not yet shot)

    # Check if action is valid
    i, j = action // 10, action % 10
    is_valid_action = (action >= 0) & (action < 100) & action_mask[i, j]

    def fire_shot() -> Tuple[EnvState, chex.Array]:
        """Fire a shot at the enemy board."""
        # Check if shot hits enemy ship
        enemy_has_ship = enemy_ship_board[i, j] < 0  # any ship type

        # Update fire board: 1 for hit, -1 for miss
        shot_result = jnp.where(enemy_has_ship, 1, -1)
        new_play_board = play_board.at[i, j].set(shot_result)

        # If hit, check if ship is completely destroyed
        def check_ship_destroyed():
            ship_type = enemy_ship_board[i, j]  # -1 to -5
            ship_index = -(ship_type + 1)  # convert to 0-4

            # Check if all positions of this ship type are hit
            ship_positions = enemy_ship_board == ship_type
            hit_positions = new_play_board == 1
            ship_destroyed = jnp.all(ship_positions <= hit_positions)

            return ship_destroyed, ship_index

        # Update alive ships if ship is destroyed
        ship_destroyed, ship_index = jax.lax.cond(
            enemy_has_ship,
            check_ship_destroyed,
            lambda: (jnp.bool_(False), jnp.int32(0))
        )

        new_alive_ships = jnp.where(
            ship_destroyed,
            state.alive_ships.at[1 - state.current_player, ship_index].set(False),
            state.alive_ships
        )

        # Check if game is over (all enemy ships destroyed)
        game_over = jnp.all(~new_alive_ships[1 - state.current_player])

        # Update winner if game is over
        new_winner = jnp.where(game_over, state.current_player, state.winner)

        # Per-step reward: +1/TOTAL_SHIP_POSITIONS for a hit, 0 for a miss
        hit_reward = jnp.where(enemy_has_ship, 1.0 / TOTAL_SHIP_POSITIONS, 0.0)
        step_reward = jnp.zeros(2, dtype=jnp.float32)
        step_reward = step_reward.at[state.current_player].set(hit_reward)
        step_reward = step_reward.at[1 - state.current_player].set(-hit_reward)

        return state.replace(
            board=state.board.at[state.current_player, 1].set(new_play_board),
            alive_ships=new_alive_ships,
            winner=new_winner
        ), step_reward

    new_state, step_reward = jax.lax.cond(
        is_valid_action,
        fire_shot,
        lambda: (state, jnp.zeros(2, dtype=jnp.float32))
    )

    return new_state, is_valid_action, step_reward


def get_play_stage_action_mask(play_board: chex.Array) -> chex.Array:
    """Get action mask for play stage.

    Args:
        play_board: (10, 10) current player's fire board

    Returns:
        (100,) boolean mask where True indicates valid shot positions (flattened)
    """
    mask_2d = play_board == 0  # positions not yet shot
    return mask_2d.flatten()  # flatten to match 1D action space
