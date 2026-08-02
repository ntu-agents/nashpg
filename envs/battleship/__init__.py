"""Battleship environment module.

This module provides a complete implementation of the Battleship game environment
for multi-agent reinforcement learning. The environment is organized into separate
modules for better maintainability:

- state.py: Environment state definitions
- setup.py: Setup stage logic (ship placement)
- gameplay.py: Play stage logic (combat mechanics)
- utils.py: Utility functions and constants

"""

from typing import Dict, Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
import envs.myspaces as env_spaces
from .state import EnvState
from .setup import setup_stage_act, setup_stage_action_mask
from .gameplay import play_stage_act, get_play_stage_action_mask
from .utils import SHIP_SIZES, SETUP_STAGE, PLAY_STAGE, get_invalid_action_rewards, get_current_ship_being_placed, is_placing_tail


class BattleShip(env_types.BaseEnv):
    """
    Battleship environment implementing the classic naval strategy game.
    
    This is a two-player turn-based game with two distinct phases:
    
    ## Game Flow
    
    ### Setup Stage (Stage 0)
    Players take turns placing their 5 ships on a 10x10 grid. Ship placement requires 2 steps:
    1. **Head Placement**: Choose where to place the ship's head position
    2. **Tail Placement**: Choose where to place the ship's tail position
    
    Ships must be placed in order: Carrier(5) → Battleship(4) → Cruiser(3) → Submarine(3) → Destroyer(2)
    Ships must be placed in straight lines (horizontal or vertical) without gaps.
    Setup ends when both players have placed all 5 ships.
    
    ### Play Stage (Stage 1)
    Players alternate taking shots at opponent's grid. The game ends when one player's
    ships are all destroyed (hit in all positions).
    
    ## Observation Space
    
    The observation contains:
    - `stage`: Current game stage (0=setup, 1=play)
    - `my_alive_ships`: Boolean array [5] indicating which of your ships are still alive
    - `enemy_alive_ships`: Boolean array [5] indicating which enemy ships are still alive
    - `board`: 10x10 grid with different meanings per stage
    
    ### Board Encoding
    
    **Setup Stage:**
    - `-1`: Occupied by Carrier (ship index 0)
    - `-2`: Occupied by Battleship (ship index 1)
    - `-3`: Occupied by Cruiser (ship index 2)
    - `-4`: Occupied by Submarine (ship index 3)
    - `-5`: Occupied by Destroyer (ship index 4)
    - `0`: Free space available for ship placement
    - `1`: Occupied by your current ship head (during tail placement)
    
    **Play Stage:**
    - `-1`: Miss (you shot here, no ship)
    - `0`: Unknown (not yet targeted)
    - `1`: Hit (you shot here, enemy ship present)
    
    ## Action Space
    
    Actions are positions on the 10x10 grid, represented as integers 0-99.
    Position `i*10 + j` corresponds to grid coordinates `(i, j)`.
    
    Valid actions are constrained by action masks (flattened to size 100):
    - **Setup Stage**: Only positions that allow valid ship placement
    - **Play Stage**: Only positions not yet targeted
    
    ## Ship Information
    
    Ships are placed in fixed order with sizes:
    - Index 0: Carrier (5 spaces)
    - Index 1: Battleship (4 spaces)
    - Index 2: Cruiser (3 spaces)
    - Index 3: Submarine (3 spaces)
    - Index 4: Destroyer (2 spaces)
    
    ## Winning Conditions
    
    A player wins when all opponent ships are completely destroyed.
    """

    def __init__(self):
        self.ship_sizes = SHIP_SIZES

    @property
    def env_name(self) -> str:
        return "battle_ship"

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def action_space(self) -> env_spaces.Discrete:
        return env_spaces.Discrete(num_categories=10*10)

    @property
    def observation_space(self) -> env_spaces.Dict:
        return env_spaces.Dict({
            "stage": env_spaces.Discrete(2, dtype=jnp.int32), # "setup" | "playing"
            
            # [Carrier(5), Battleship(4), Cruiser(3), Submarine(3), Destroyer(2)]
            "my_alive_ships": env_spaces.MultiDiscrete([2, 2, 2, 2, 2], dtype=jnp.bool),
            "enemy_alive_ships": env_spaces.MultiDiscrete([2, 2, 2, 2, 2], dtype=jnp.bool),
            # Setup Stage:
            #     -1 to -5 => occupied by ships (Carrier to Destroyer)
            #     0 => free space
            #     1 => occupied by your current ship head
            # Play Stage:
            #     -1 => miss
            #     0 => nothing yet
            #     1 => hit
            "board": env_spaces.Box(low=-5, high=1, shape=(10, 10), dtype=jnp.int32)
        })
    
    
    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, env_types.TimeStep]:
        """Reset the environment to initial state.
        
        Args:
            key: JAX random key for initialization
            
        Returns:
            Tuple of (initial_state, initial_timestep)
        """
        key, subkey = jax.random.split(key)
        current_player = jax.random.bernoulli(subkey, p=0.5).astype(dtype=jnp.int32)

        state = EnvState(
            key=key,
            current_player=current_player,
            done=jnp.bool(False),
            step_cnt=jnp.int32(0),
            stage=jnp.int32(SETUP_STAGE),
            alive_ships=jnp.ones((2, 5), dtype=jnp.bool),
            board=jnp.zeros((2, 2, 10, 10), dtype=jnp.int32),
            winner=jnp.int32(-1)
        )

        timestep = env_types.TimeStep(
            reward=jnp.zeros((2, ), dtype=jnp.float32),
            done=jnp.bool(False),
            observation={
                "stage": state.stage,
                "my_alive_ships": state.alive_ships[state.current_player],
                "enemy_alive_ships": state.alive_ships[1 - state.current_player],
                "board": state.board[state.current_player, state.stage],
            },
            action_mask=jnp.ones((100,), dtype=jnp.bool),
            current_player=state.current_player,
            info={"step_cnt": state.step_cnt}
        )

        return state, timestep
    

    @partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action: env_types.Action) -> Tuple[EnvState, env_types.TimeStep]:
        """Execute one step of the environment.
        
        Args:
            state: Current environment state
            action: Action to take (position 0-99)
            
        Returns:
            Tuple of (new_state, timestep)
        """
        
        # Capture done status before any modification so reward logic can use it
        was_done = state.done

        state, is_valid_action, step_reward = jax.lax.cond(
            ~state.done,
            lambda s, a: jax.lax.cond(
                state.stage == SETUP_STAGE,
                lambda s, a: setup_stage_act(s, a),
                lambda s, a: play_stage_act(s, a),
                s, a
            ),
            lambda s, _: (s, jnp.bool(False), jnp.zeros(2, dtype=jnp.float32)),
            state, action
        )

        # Update states
        state = jax.lax.cond(
            ~was_done,
            lambda: jax.lax.cond(
                is_valid_action,
                lambda: state.replace(
                    done = state.winner != -1,
                    step_cnt = state.step_cnt + 1,
                    current_player = 1 - state.current_player
                ),
                lambda: state.replace(
                    done = jnp.bool(True),
                    winner = 1 - state.current_player,
                    step_cnt = state.step_cnt + 1,
                    current_player = 1 - state.current_player
                )
            ),
            lambda: state,
        )
        
        

        # Create new timestep
        def get_observation_board():
            return state.board[state.current_player, state.stage]
        
        def get_action_mask():
            # Setup stage: use ship placement action mask
            def setup_mask():
                ship_board = state.board[state.current_player, 0]
                current_ship = get_current_ship_being_placed(ship_board)
                is_putting_tail = is_placing_tail(ship_board)
                return setup_stage_action_mask(ship_board, current_ship, is_putting_tail)
            
            # Play stage: positions not yet shot
            def play_mask():
                play_board = state.board[state.current_player, 1]
                return get_play_stage_action_mask(play_board)
            
            return jax.lax.cond(
                state.stage == SETUP_STAGE,
                setup_mask,
                play_mask
            )
        
        # Reward logic:
        #   - Normal valid action: use per-step hit reward (+1/17 per hit, 0 for miss/setup)
        #   - Invalid action on live game: terminal ±1 (opponent wins)
        #   - Already-done game: zeros (was_done=True => is_valid_action=False, caught by first cond)
        reward = jax.lax.cond(
            was_done | is_valid_action,
            lambda: step_reward,
            lambda: get_invalid_action_rewards(state.winner),
        )

        timestep = env_types.TimeStep(
            reward=reward,
            done=state.done,
            observation={
                "stage": state.stage,
                "my_alive_ships": state.alive_ships[state.current_player],
                "enemy_alive_ships": state.alive_ships[1 - state.current_player],
                "board": get_observation_board(),
            },
            action_mask=get_action_mask(),
            current_player=state.current_player,
            info={"step_cnt": state.step_cnt}
        )
        
        return state, timestep


# Export the main class and state for external use
__all__ = ["BattleShip", "EnvState"]