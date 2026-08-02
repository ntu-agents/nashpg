"""State definitions for Battleship environment."""

import chex


@chex.dataclass
class EnvState:
    """State for the Battleship environment.
    
    Attributes:
        key: JAX random key for randomness
        current_player: Current player (0 or 1)
        done: Whether the game is finished
        step_cnt: Number of steps taken
        stage: Current game stage (0=setup, 1=play)
        alive_ships: Boolean array (2, 5) indicating which ships are alive for each player
        board: Game board array (2, 2, 10, 10) - (player_id, board_type, row, col)
               board_type: 0=ship_board, 1=fire_board
        winner: Winner of the game (-1 if no winner yet)
    """
    # Base environment state
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    
    # Battleship-specific state
    stage: chex.Array
    alive_ships: chex.Array  # (2, 5) (player_id, ship_index)
    board: chex.Array  # (2, 2, 10, 10) (player_id, ship_board | fire_board, row, col)
    winner: chex.Numeric