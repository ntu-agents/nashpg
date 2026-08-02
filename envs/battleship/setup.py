"""Setup stage logic for Battleship environment."""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
from .state import EnvState


def setup_stage_act(
    state: EnvState,
    action: env_types.Action
) -> Tuple[EnvState, chex.Numeric, chex.Array]:
    """Perform action at setup stage.

    Args:
        state: Current environment state
        action: Action to take (position 0-99)
    Returns:
        Tuple of (new_state, is_valid_action, step_reward)
        step_reward is always zeros (no reward during setup)
    """
    ship_board = state.board[state.current_player, 0]  # this player ship board

    # Calculate what the current ship is, and whether we're putting head or tail
    accum_ship_sizes = jnp.array([0, 5, 9, 12, 15], dtype=jnp.int32)  # ship_sizes [5, 4, 3, 3, 2]
    total_size_on_board = jnp.sum(ship_board < 0)  # Count occupied spaces by all ships
    current_ship = jnp.argmax(accum_ship_sizes == total_size_on_board)
    is_putting_tail = jnp.any(ship_board == 1)

    # Check valid action
    i, j = action // 10, action % 10
    action_mask = setup_stage_action_mask(ship_board, current_ship, is_putting_tail)
    is_valid_action = (action >= 0) & (action < 100) & (action_mask[action] == True)

    def put_head() -> EnvState:
        """Place ship head at the action position."""
        return state.replace(
            board=state.board.at[state.current_player, 0, i, j].set(1)
        )

    def put_tail() -> EnvState:
        """Place ship tail and complete the ship."""
        # Find the head position (marked with 1) - exactly one position has value 1
        head_flat_idx = jnp.argmax(ship_board == 1)
        head_i, head_j = head_flat_idx // 10, head_flat_idx % 10
        
        # Determine orientation
        is_horizontal = (i == head_i)
        
        def fill_ship_size(size: int):
            """Fill ship positions based on size."""
            # For horizontal placement
            def fill_horizontal():
                start_j = jnp.minimum(head_j, j)
                # Create static indices based on size
                all_indices = jnp.arange(10)
                positions_j = jax.lax.dynamic_slice(all_indices, (start_j,), (size,))
                positions_i = jnp.full(size, i)
                return positions_i, positions_j
            
            # For vertical placement
            def fill_vertical():
                start_i = jnp.minimum(head_i, i)
                # Create static indices based on size
                all_indices = jnp.arange(10)
                positions_i = jax.lax.dynamic_slice(all_indices, (start_i,), (size,))
                positions_j = jnp.full(size, j)
                return positions_i, positions_j
            
            # Get all positions to fill
            fill_i, fill_j = jax.lax.cond(
                is_horizontal,
                fill_horizontal,
                fill_vertical
            )
            
            # Fill all positions with ship type marker (-(current_ship + 1))
            ship_marker = -(current_ship + 1)  # -1 for Carrier, -2 for Battleship, etc.
            return ship_board.at[fill_i, fill_j].set(ship_marker)
        
        # Use switch to handle each ship size statically
        final_ship_board = jax.lax.switch(
            current_ship,
            [
                lambda: fill_ship_size(5),  # Carrier
                lambda: fill_ship_size(4),  # Battleship
                lambda: fill_ship_size(3),  # Cruiser
                lambda: fill_ship_size(3),  # Submarine
                lambda: fill_ship_size(2),  # Destroyer
            ]
        )

        # Update board with completed ship
        updated_board = state.board.at[state.current_player, 0].set(final_ship_board)
        
        # Check if both players have placed all ships (17 positions each)
        player0_ships_complete = jnp.sum(updated_board[0, 0] < 0) == 17
        player1_ships_complete = jnp.sum(updated_board[1, 0] < 0) == 17
        both_players_complete = player0_ships_complete & player1_ships_complete
        
        # Update stage to play stage (1) if both players completed setup
        new_stage = jnp.where(both_players_complete, 1, state.stage)
        
        return state.replace(
            board=updated_board,
            stage=new_stage
        )

    new_state = jax.lax.cond(
        is_valid_action,
        lambda: jax.lax.cond(is_putting_tail, put_tail, put_head),
        lambda: state,  # Return unchanged state for invalid actions
    )

    return new_state, is_valid_action, jnp.zeros(2, dtype=jnp.float32)


def setup_stage_action_mask(
    ship_board: chex.Array, 
    current_ship: chex.Numeric, 
    put_tail: chex.Numeric
) -> chex.Array:
    """Calculate action mask for setup stage ship placement.
    
    During setup, players place ships by first placing the head, then the tail.
    This function determines which positions are valid for the current placement step.
    
    Args:
        ship_board: (10, 10) array representing current player's ship board
                   -1 = occupied by previous ships, 0 = empty, 1 = current ship head
        current_ship: Index of current ship being placed (0-4)
        put_tail: Boolean indicating if placing tail (True) or head (False)
        ship_sizes: Array of ship sizes [5, 4, 3, 3, 2]
        
    Returns:
        (100,) boolean array where True indicates valid action positions (flattened)
    """
    
    def put_head_action_mask(ship_board: chex.Array, ship_size: int) -> chex.Array:
        """Generate action mask for placing ship head.
        
        A position is valid for head placement if:
        1. The position is empty (not occupied by previous ships)
        2. There exists at least one valid tail position in any of the 4 directions
        
        Args:
            ship_board: (10, 10) current ship board state
            ship_size: Size of ship being placed (static for JAX compilation)
            
        Returns:
            (10, 10) boolean mask of valid head positions
        """
        
        def check_position(i, j):
            """Check if position (i,j) is valid for head placement."""
            # Position is invalid if already occupied (any non-zero value)
            invalid_pos = ship_board[i, j] != 0
            
            # Calculate slice size
            slice_size = ship_size - 1
            
            # Check if we can place tail in horizontal directions
            can_place_horizontally = (
                # Can place tail to the right: head at (i,j), tail at (i, j+ship_size-1)
                ((j + slice_size < 10) & jnp.all(jax.lax.dynamic_slice(ship_board[i], (j + 1,), (slice_size,)) == 0)) |
                # Can place tail to the left: head at (i,j), tail at (i, j-ship_size+1)
                ((j - slice_size >= 0) & jnp.all(jax.lax.dynamic_slice(ship_board[i], (j - slice_size,), (slice_size,)) == 0))
            )
            
            # Check if we can place tail in vertical directions
            can_place_vertically = (
                # Can place tail downward: head at (i,j), tail at (i+ship_size-1, j)
                ((i + slice_size < 10) & jnp.all(jax.lax.dynamic_slice(ship_board[:, j], (i + 1,), (slice_size,)) == 0)) |
                # Can place tail upward: head at (i,j), tail at (i-ship_size+1, j)
                ((i - slice_size >= 0) & jnp.all(jax.lax.dynamic_slice(ship_board[:, j], (i - slice_size,), (slice_size,)) == 0))
            )
            
            return ~invalid_pos & (can_place_horizontally | can_place_vertically)
        
        # Create action mask for all positions using vectorized check
        i_coords, j_coords = jnp.meshgrid(jnp.arange(10), jnp.arange(10), indexing='ij')
        action_mask = jax.vmap(jax.vmap(check_position))(i_coords, j_coords)
        
        return action_mask
    
    def put_tail_action_mask(ship_board: chex.Array, ship_size: int) -> chex.Array:
        """Generate action mask for placing ship tail.
        
        Given a head position (marked with 1), validates the 4 possible tail positions:
        - Right: (head_i, head_j + ship_size - 1)
        - Left:  (head_i, head_j - ship_size + 1)
        - Down:  (head_i + ship_size - 1, head_j)
        - Up:    (head_i - ship_size + 1, head_j)
        
        A tail position is valid if:
        1. It's within board bounds
        2. All positions between head and tail are empty
        
        Args:
            ship_board: (10, 10) current ship board state with head marked as 1
            ship_size: Size of ship being placed (static for JAX compilation)
            
        Returns:
            (10, 10) boolean mask with exactly 0-4 True values at valid tail positions
        """
        
        # Find the head position (marked with 1) - exactly one position has value 1
        head_flat_idx = jnp.argmax(ship_board == 1)
        head_i, head_j = head_flat_idx // 10, head_flat_idx % 10
        
        # Initialize action mask to all False - only valid tail positions will be True
        action_mask = jnp.zeros((10, 10), dtype=jnp.bool)
        
        # Calculate slice size (ship_size - 1)
        slice_size = ship_size - 1
        
        # Check tail position to the right
        tail_right_j = head_j + slice_size
        right_in_bounds = (tail_right_j < 10) & (head_j + 1 < 10) & (head_j + slice_size < 10)
        right_slice = jax.lax.dynamic_slice(ship_board[head_i], (head_j + 1,), (slice_size,))
        right_valid = right_in_bounds & jnp.all(right_slice == 0)
        action_mask = action_mask.at[head_i, tail_right_j].set(right_valid)
        
        # Check tail position to the left
        tail_left_j = head_j - slice_size
        left_in_bounds = (tail_left_j >= 0) & (tail_left_j < head_j)
        left_slice = jax.lax.dynamic_slice(ship_board[head_i], (tail_left_j,), (slice_size,))
        left_valid = left_in_bounds & jnp.all(left_slice == 0)
        action_mask = action_mask.at[head_i, tail_left_j].set(left_valid)
        
        # Check tail position downward
        tail_down_i = head_i + slice_size
        down_in_bounds = (tail_down_i < 10) & (head_i + 1 < 10) & (head_i + slice_size < 10)
        down_slice = jax.lax.dynamic_slice(ship_board[:, head_j], (head_i + 1,), (slice_size,))
        down_valid = down_in_bounds & jnp.all(down_slice == 0)
        action_mask = action_mask.at[tail_down_i, head_j].set(down_valid)
        
        # Check tail position upward
        tail_up_i = head_i - slice_size
        up_in_bounds = (tail_up_i >= 0) & (tail_up_i < head_i)
        up_slice = jax.lax.dynamic_slice(ship_board[:, head_j], (tail_up_i,), (slice_size,))
        up_valid = up_in_bounds & jnp.all(up_slice == 0)
        action_mask = action_mask.at[tail_up_i, head_j].set(up_valid)
        
        return action_mask
    
    
    # Switch based on current_ship with static ship sizes for JAX compilation
    def carrier_mask(board):
        return jax.lax.cond(put_tail, 
                           lambda b: put_tail_action_mask(b, 5), 
                           lambda b: put_head_action_mask(b, 5), 
                           board)
    
    def battleship_mask(board):
        return jax.lax.cond(put_tail, 
                           lambda b: put_tail_action_mask(b, 4), 
                           lambda b: put_head_action_mask(b, 4), 
                           board)
                           
    def cruiser_mask(board):
        return jax.lax.cond(put_tail, 
                           lambda b: put_tail_action_mask(b, 3), 
                           lambda b: put_head_action_mask(b, 3), 
                           board)
                           
    def submarine_mask(board):
        return jax.lax.cond(put_tail, 
                           lambda b: put_tail_action_mask(b, 3), 
                           lambda b: put_head_action_mask(b, 3), 
                           board)
                           
    def destroyer_mask(board):
        return jax.lax.cond(put_tail, 
                           lambda b: put_tail_action_mask(b, 2), 
                           lambda b: put_head_action_mask(b, 2), 
                           board)
    
    mask_2d = jax.lax.switch(
        current_ship,
        [carrier_mask, battleship_mask, cruiser_mask, submarine_mask, destroyer_mask],
        ship_board
    )
    return mask_2d.flatten()  # flatten to match 1D action space