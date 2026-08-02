"""Phantom Tic-Tac-Toe environment implementation using JAX.

Observation matches OpenSpiel's information_state_tensor (108-dim):
  [0:27]   board view: 3-state one-hot per cell (absolute encoding)
               bits [0:9]   = unknown cells
               bits [9:18]  = player 1 (kNought) pieces
               bits [18:27] = player 0 (kCross) pieces
  [27:108] action history: player's own attempted actions (successful + failed),
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


@chex.dataclass
class EnvState:
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    true_board: chex.Array        # (3, 3)  -1=empty, 0=player0, 1=player1
    player_knowledge: chex.Array  # (2, 3, 3)  -1=unknown, 0=P0 stone, 1=P1 stone (absolute)
    winner: chex.Numeric          # -1=none, 0=p0, 1=p1, 2=draw
    is_abrupt_version: chex.Numeric
    action_history: chex.Array    # (2, 9) int32  own attempted actions per player, -1=unused slot


class PhantomTicTacToe(env_types.BaseEnv):
    """Phantom Tic-Tac-Toe: imperfect-information TicTacToe.

    Observation (108-dim) = board view (27) + action history (81),
    matching OpenSpiel's information_state_tensor for phantom_ttt.
    """

    def __init__(self, is_abrupt: bool = False):
        self.is_abrupt = is_abrupt

    @property
    def env_name(self) -> str:
        return f"phantom_tic_tac_toe_{'abrupt' if self.is_abrupt else 'classic'}"

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def action_space(self) -> Discrete:
        return Discrete(num_categories=9)

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(108,), dtype=jnp.float32)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, env_types.TimeStep]:
        key, player_key = jax.random.split(key)
        starting_player = jax.random.bernoulli(player_key).astype(jnp.int32)

        initial_state = EnvState(
            key=key,
            current_player=starting_player,
            done=jnp.bool_(False),
            step_cnt=jnp.int32(0),
            true_board=-jnp.ones((3, 3), dtype=jnp.int32),
            player_knowledge=-jnp.ones((2, 3, 3), dtype=jnp.int32),
            winner=jnp.int32(-1),
            is_abrupt_version=jnp.bool_(self.is_abrupt),
            action_history=jnp.full((2, 9), -1, dtype=jnp.int32),
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
        row, col = action // 3, action % 3
        is_valid_position = (action >= 0) & (action < 9)

        updated_state = jax.lax.cond(
            ~state.done,
            lambda s: jax.lax.cond(
                is_valid_position,
                partial(self._process_move, row, col, action),
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

    def _process_move(self, row, col, action, state: EnvState) -> EnvState:
        position_is_empty = state.true_board[row, col] == -1
        return jax.lax.cond(
            position_is_empty,
            partial(self._successful_placement, row, col, action),
            partial(self._blocked_placement, row, col, action),
            state
        )

    def _record_action(self, state: EnvState, player, action) -> chex.Array:
        """Append action to player's action_history in the next free slot."""
        slot = jnp.sum(state.action_history[player] >= 0)
        return state.action_history.at[player, slot].set(action)

    def _successful_placement(self, row, col, action, state: EnvState) -> EnvState:
        current_player = state.current_player
        updated_board = state.true_board.at[row, col].set(current_player)
        updated_knowledge = state.player_knowledge.at[current_player, row, col].set(current_player)
        updated_history = self._record_action(state, current_player, action)
        has_won = self._check_winner(updated_board, current_player)
        is_draw = jnp.all(updated_board != -1)
        return state.replace(
            true_board=updated_board,
            player_knowledge=updated_knowledge,
            action_history=updated_history,
            winner=jnp.where(has_won, current_player,
                             jnp.where(is_draw, jnp.int32(2), jnp.int32(-1))),
            done=has_won | is_draw,
            step_cnt=state.step_cnt + 1,
            current_player=1 - current_player,
        )

    def _blocked_placement(self, row, col, action, state: EnvState) -> EnvState:
        current_player = state.current_player
        updated_knowledge = state.player_knowledge.at[current_player, row, col].set(1 - current_player)
        updated_history = self._record_action(state, current_player, action)
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
        return state.replace(
            winner=1 - state.current_player,
            done=jnp.bool_(True),
            step_cnt=state.step_cnt + 1,
            current_player=1 - state.current_player,
        )

    def _build_player_observation(self, knowledge_flat, history) -> chex.Array:
        """Build 108-dim infostate from a player's (9,) knowledge and (9,) history.

        Layout matches OpenSpiel information_state_tensor (absolute encoding):
          [0:9]   = unknown cells (knowledge == -1)
          [9:18]  = player 1 / kNought pieces (knowledge == 1)
          [18:27] = player 0 / kCross pieces (knowledge == 0)
          [27:108]= action history: 9 slots × 9-bit one-hot
        """
        # 27-dim board view: map -1→0 (unknown), 1→1 (P1 stone), 0→2 (P0 stone)
        state_idx = jnp.where(knowledge_flat == -1, 0,
                              jnp.where(knowledge_flat == 1, 1, 2))
        obs_board = jax.nn.one_hot(state_idx, 3).T.reshape(-1)  # (27,)

        # 81-dim action history: one-hot per slot, zero for unused slots
        valid = history >= 0
        hist_clamped = jnp.where(valid, history, 0)
        obs_hist = jnp.where(
            valid[:, None],
            jax.nn.one_hot(hist_clamped, 9),
            jnp.zeros((9, 9))
        ).reshape(-1)  # (81,)

        return jnp.concatenate([obs_board, obs_hist])  # (108,)

    def _get_observation(self, state: EnvState) -> chex.Array:
        """108-dim infostate for the current player."""
        p = state.current_player
        knowledge_flat = state.player_knowledge[p].reshape(-1)
        history = state.action_history[p]
        return self._build_player_observation(knowledge_flat, history)

    def _get_observation_for_player(self, state: EnvState, player: int) -> chex.Array:
        """108-dim infostate for any player (used in testing/exploitability)."""
        knowledge_flat = state.player_knowledge[player].reshape(-1)
        history = state.action_history[player]
        return self._build_player_observation(knowledge_flat, history)

    def _get_action_mask(self, state: EnvState) -> chex.Array:
        current_knowledge = state.player_knowledge[state.current_player]
        return (current_knowledge == -1).flatten()

    def _check_winner(self, board: chex.Array, player: chex.Numeric) -> chex.Numeric:
        horizontal = jnp.any(jnp.all(board == player, axis=1))
        vertical = jnp.any(jnp.all(board == player, axis=0))
        diag_main = jnp.all(jnp.diag(board) == player)
        diag_anti = jnp.all(jnp.diag(jnp.fliplr(board)) == player)
        return horizontal | vertical | diag_main | diag_anti

    def _calculate_rewards(self, state: EnvState) -> chex.Array:
        return jnp.where(
            state.winner == -1,
            jnp.zeros(2, dtype=jnp.float32),
            jnp.where(
                state.winner == 2,
                jnp.zeros(2, dtype=jnp.float32),
                jnp.where(
                    state.winner == 0,
                    jnp.array([1.0, -1.0], dtype=jnp.float32),
                    jnp.array([-1.0, 1.0], dtype=jnp.float32),
                )
            )
        )
