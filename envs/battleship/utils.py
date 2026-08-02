"""Utility functions and constants for Battleship environment."""

import jax.numpy as jnp
import chex


# Ship configuration constants
SHIP_SIZES = jnp.array([5, 4, 3, 3, 2])  # [Carrier, Battleship, Cruiser, Submarine, Destroyer]
SHIP_NAMES = ["Carrier", "Battleship", "Cruiser", "Submarine", "Destroyer"]
TOTAL_SHIP_POSITIONS = 17  # Sum of all ship sizes

# Board configuration
BOARD_SIZE = 10
TOTAL_POSITIONS = BOARD_SIZE * BOARD_SIZE

# Stage constants
SETUP_STAGE = 0
PLAY_STAGE = 1

# Board encoding values
EMPTY = 0
SHIP_HEAD = 1
MISS = -1
HIT = 1

# Ship type markers (negative values)
CARRIER_MARKER = -1
BATTLESHIP_MARKER = -2
CRUISER_MARKER = -3
SUBMARINE_MARKER = -4
DESTROYER_MARKER = -5


def count_ships_placed(ship_board: chex.Array) -> chex.Numeric:
    return jnp.sum(ship_board < 0)


def get_current_ship_being_placed(ship_board: chex.Array) -> chex.Numeric:
    """argmax is safe here since accum_ship_sizes always contains total_size_on_board."""
    accum_ship_sizes = jnp.array([0, 5, 9, 12, 15], dtype=jnp.int32)
    total_size_on_board = count_ships_placed(ship_board)
    return jnp.argmax(accum_ship_sizes == total_size_on_board)


def is_placing_tail(ship_board: chex.Array) -> chex.Numeric:
    return jnp.any(ship_board == SHIP_HEAD)


def get_invalid_action_rewards(winner: chex.Numeric) -> chex.Array:
    """Terminal rewards when a player makes an invalid action.

    Only called for invalid-action terminations — NOT for normal wins,
    which are handled by accumulated per-hit rewards (+1/TOTAL_SHIP_POSITIONS each).
    """
    rewards = jnp.zeros(2, dtype=jnp.float32)
    rewards = rewards.at[winner].set(1.0)
    rewards = rewards.at[1 - winner].set(-1.0)
    return rewards
