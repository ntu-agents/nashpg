"""Dark Hex 3 environment implementation using JAX.

Observation matches OpenSpiel's information_state_tensor (162-dim):
  [0:81]   board view: 9-state one-hot per cell (9 cells × 9 categories)
               cell i's category is at bits [9*i : 9*i+9]
               col 4 = unknown, col 5 = player 0 stone (absolute), col 3 = player 1 stone
  [81:162] action history: player's own attempted actions (successful + failed),
               9 slots × 9-bit one-hot (padded with zeros if fewer than 9 actions)

Two versions:
- Classic: When a move is blocked, player can try again on the same turn
- Abrupt:  When a move is blocked, turn immediately ends
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
from envs.myspaces import Discrete, Box

# Board topology for 3x3 hexagonal grid
# Positions:  0 1 2
#              3 4 5
#               6 7 8
NEIGHBORS = jnp.array([
    [1, 3, -1, -1, -1, -1],
    [0, 2, 3, 4, -1, -1],
    [1, 4, 5, -1, -1, -1],
    [0, 1, 4, 6, -1, -1],
    [1, 2, 3, 5, 6, 7],
    [2, 4, 7, 8, -1, -1],
    [3, 4, 7, -1, -1, -1],
    [4, 5, 6, 8, -1, -1],
    [5, 7, -1, -1, -1, -1],
], dtype=jnp.int32)

# Player 0: connects top edge {0,1,2} to bottom edge {6,7,8}
# Player 1: connects left edge {0,3,6} to right edge {2,5,8}
PLAYER0_WINNING_PATHS = jnp.array([
    [0, 3, 6, -1, -1],
    [1, 3, 6, -1, -1],
    [1, 4, 6, -1, -1],
    [1, 4, 7, -1, -1],
    [2, 4, 6, -1, -1],
    [2, 4, 7, -1, -1],
    [2, 5, 7, -1, -1],
    [2, 5, 8, -1, -1],
    [0, 3, 4, 7, -1],
    [1, 4, 5, 8, -1],
    [0, 3, 4, 5, 8],
], dtype=jnp.int32)

PLAYER1_WINNING_PATHS = jnp.array([
    [0, 1, 2, -1, -1],
    [3, 1, 2, -1, -1],
    [3, 4, 2, -1, -1],
    [3, 4, 5, -1, -1],
    [6, 4, 2, -1, -1],
    [6, 4, 5, -1, -1],
    [6, 7, 5, -1, -1],
    [6, 7, 8, -1, -1],
    [0, 1, 4, 5, -1],
    [3, 4, 7, 8, -1],
    [0, 1, 4, 7, 8],
], dtype=jnp.int32)


@chex.dataclass
class EnvState:
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    true_board: chex.Array        # (9,)  -1=empty, 0=game-P0 stone, 1=game-P1 stone
    player_knowledge: chex.Array  # (2, 9)  indexed by agent; values: -1=unknown, 0=P0 stone, 1=P1 stone
    winner: chex.Numeric          # -1=none, 0=game-P0 won, 1=game-P1 won
    is_abrupt_version: chex.Numeric
    action_history: chex.Array    # (2, 9) int32  own attempted actions per agent, -1=unused slot
    starting_agent: chex.Numeric  # agent assigned game-P0 this episode (goes first)


class DarkHex3(env_types.BaseEnv):
    """Dark Hex 3: imperfect-information Hex on a 3×3 grid.

    Observation (162-dim) = board view (81) + action history (81),
    matching OpenSpiel's information_state_tensor for dark_hex(board_size=3).
    """

    def __init__(self, is_abrupt: bool = False):
        self.is_abrupt = is_abrupt

    @property
    def env_name(self) -> str:
        return f"dark_hex3_{'abrupt' if self.is_abrupt else 'classic'}"

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def action_space(self) -> Discrete:
        return Discrete(num_categories=9)

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(162,), dtype=jnp.float32)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, env_types.TimeStep]:
        key, agent_key = jax.random.split(key)
        starting_agent = jax.random.bernoulli(agent_key).astype(jnp.int32)

        initial_state = EnvState(
            key=key,
            current_player=starting_agent,  # game-P0's agent always moves first
            done=jnp.bool_(False),
            step_cnt=jnp.int32(0),
            true_board=-jnp.ones((9,), dtype=jnp.int32),
            player_knowledge=-jnp.ones((2, 9), dtype=jnp.int32),
            winner=jnp.int32(-1),
            is_abrupt_version=jnp.bool_(self.is_abrupt),
            action_history=jnp.full((2, 9), -1, dtype=jnp.int32),
            starting_agent=starting_agent,
        )

        initial_timestep = env_types.TimeStep(
            reward=jnp.zeros((2,), dtype=jnp.float32),
            done=initial_state.done,
            observation=self._get_observation(initial_state),
            action_mask=self._get_action_mask(initial_state),
            current_player=initial_state.current_player,
            info={"step_cnt": initial_state.step_cnt}
        )

        return initial_state, initial_timestep

    @partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action: env_types.Action) -> Tuple[EnvState, env_types.TimeStep]:
        chex.assert_shape(action.shape, ())
        is_valid_position = (action >= 0) & (action < 9)

        updated_state = jax.lax.cond(
            ~state.done,
            lambda s: jax.lax.cond(
                is_valid_position,
                partial(self._process_move, action),
                self._handle_invalid_position,
                s
            ),
            lambda s: s,
            state
        )

        rewards = self._calculate_rewards(updated_state)

        final_timestep = env_types.TimeStep(
            reward=rewards,
            done=updated_state.done,
            observation=self._get_observation(updated_state),
            action_mask=self._get_action_mask(updated_state),
            current_player=updated_state.current_player,
            info={"step_cnt": updated_state.step_cnt}
        )

        return updated_state, final_timestep

    def _process_move(self, position, state: EnvState) -> EnvState:
        position_is_empty = state.true_board[position] == -1
        return jax.lax.cond(
            position_is_empty,
            partial(self._successful_placement, position),
            partial(self._blocked_placement, position),
            state
        )

    def _record_action(self, state: EnvState, player, action) -> chex.Array:
        """Append action to player's action_history in the next free slot."""
        slot = jnp.sum(state.action_history[player] >= 0)
        return state.action_history.at[player, slot].set(action)

    def _successful_placement(self, position, state: EnvState) -> EnvState:
        current_player = state.current_player
        game_player = (current_player + state.starting_agent) % 2
        updated_board = state.true_board.at[position].set(game_player)
        updated_knowledge = state.player_knowledge.at[current_player, position].set(game_player)
        updated_history = self._record_action(state, current_player, position)
        has_won = self._check_winner(updated_board, game_player)
        return state.replace(
            true_board=updated_board,
            player_knowledge=updated_knowledge,
            action_history=updated_history,
            winner=jnp.where(has_won, game_player, jnp.int32(-1)),
            done=has_won,
            step_cnt=state.step_cnt + 1,
            current_player=1 - current_player,
        )

    def _blocked_placement(self, position, state: EnvState) -> EnvState:
        current_player = state.current_player
        opponent_game_player = 1 - (current_player + state.starting_agent) % 2
        updated_knowledge = state.player_knowledge.at[current_player, position].set(opponent_game_player)
        updated_history = self._record_action(state, current_player, position)
        next_player = jax.lax.cond(
            state.is_abrupt_version,
            lambda: 1 - current_player,
            lambda: current_player,
        )
        return state.replace(
            player_knowledge=updated_knowledge,
            action_history=updated_history,
            step_cnt=state.step_cnt + 1,
            current_player=next_player,
        )

    def _handle_invalid_position(self, state: EnvState) -> EnvState:
        opponent_game_player = 1 - (state.current_player + state.starting_agent) % 2
        return state.replace(
            winner=opponent_game_player,
            done=jnp.bool_(True),
            step_cnt=state.step_cnt + 1,
            current_player=1 - state.current_player,
        )

    def _build_player_observation(self, knowledge, history) -> chex.Array:
        """Build 162-dim infostate from a player's (9,) knowledge and (9,) history.

        Layout matches OpenSpiel information_state_tensor (absolute encoding):
          cell i bits [9*i : 9*i+9]: one-hot over 9 categories
            col 4 = unknown, col 5 = player 0 stone, col 3 = player 1 stone
          [81:162]: action history, 9 slots × 9-bit one-hot
        """
        # 81-dim board view: unknown(-1)→4, p0 stone(0)→5, p1 stone(1)→3
        state_idx = jnp.where(knowledge == -1, 4,
                              jnp.where(knowledge == 0, 5, 3))
        obs_board = jax.nn.one_hot(state_idx, 9).reshape(-1)  # (81,)

        # 81-dim action history
        valid = history >= 0
        hist_clamped = jnp.where(valid, history, 0)
        obs_hist = jnp.where(
            valid[:, None],
            jax.nn.one_hot(hist_clamped, 9),
            jnp.zeros((9, 9))
        ).reshape(-1)  # (81,)

        return jnp.concatenate([obs_board, obs_hist])  # (162,)

    def _get_observation(self, state: EnvState) -> chex.Array:
        """162-dim infostate for the current player."""
        p = state.current_player
        return self._build_player_observation(state.player_knowledge[p],
                                              state.action_history[p])

    def _get_observation_for_player(self, state: EnvState, player: int) -> chex.Array:
        """162-dim infostate for any player (used in testing/exploitability)."""
        return self._build_player_observation(state.player_knowledge[player],
                                              state.action_history[player])

    def _get_action_mask(self, state: EnvState) -> chex.Array:
        current_knowledge = state.player_knowledge[state.current_player]
        return (current_knowledge == -1)  # -1 = unknown = legal to play

    def _check_winner(self, board: chex.Array, player: chex.Numeric) -> chex.Numeric:
        player_paths = jax.lax.cond(
            player == 0,
            lambda: PLAYER0_WINNING_PATHS,
            lambda: PLAYER1_WINNING_PATHS
        )

        def check_path(path):
            valid_positions = path >= 0
            return jnp.all(jnp.where(valid_positions, board[path] == player, True))

        return jnp.any(jax.vmap(check_path)(player_paths))

    def _calculate_rewards(self, state: EnvState) -> chex.Array:
        # winner is a game-player index; remap to agent-indexed rewards via starting_agent.
        # agent i is game-player (i + starting_agent) % 2, so agent i wins iff winner == (i + starting_agent) % 2.
        agent0_reward = jnp.where(state.winner == state.starting_agent % 2, 1.0, -1.0)
        agent1_reward = jnp.where(state.winner == (1 + state.starting_agent) % 2, 1.0, -1.0)
        terminal_rewards = jnp.array([agent0_reward, agent1_reward], dtype=jnp.float32)
        return jnp.where(state.winner == -1, jnp.zeros(2, dtype=jnp.float32), terminal_rewards)
