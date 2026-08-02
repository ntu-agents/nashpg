"""
RNAD loss functions: V-trace and NERD policy gradient.
Adapted from https://arxiv.org/abs/2206.15378 for our framework.

Key differences from the reference:
- No OpenSpiel dependency; works with our env/agent API
- Episode boundaries handled via `is_new_eps` flag instead of a `valid` mask:
  at episode starts the backward carry is reset so no cross-episode information
  leaks, but all steps still contribute gradients (no data wasted)
- IS correction for multiple gradient epochs per batch
"""

from typing import Tuple, Any
import jax
import jax.numpy as jnp
from jax import lax
import chex


def legal_log_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
    """Log policy for legal actions, 0 for illegal actions.

    Works correctly with either raw or masked (-inf) logits.
    """
    logits_masked = logits + jnp.log(legal_actions.astype(jnp.float32))
    max_legal = logits_masked.max(axis=-1, keepdims=True)
    logits_masked = logits_masked - max_legal
    exp_logits = jnp.exp(logits_masked)
    baseline = jnp.log(jnp.sum(exp_logits, axis=-1, keepdims=True))
    # Use where to avoid NaN from illegal logits multiplied by 0
    return jnp.where(legal_actions, logits - max_legal - baseline, 0.0)


def _player_others(player_ids: chex.Array, player: int) -> chex.Array:
    """Returns +1 for steps where `player` acts, -1 for opponent steps.

    Args:
        player_ids: shape [...], integer player ids
        player: the player index

    Returns:
        shape [..., 1]
    """
    sign = 2 * (player_ids == player).astype(jnp.int32) - 1
    return jnp.expand_dims(sign, axis=-1)


def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array) -> chex.Array:
    """Returns pi(a) / mu(a) for the selected action.

    Args:
        pi: policy probabilities [..., A]
        mu: behavior policy probabilities [..., A]
        actions_oh: one-hot action [..., A]

    Returns:
        IS ratio [...]
    """
    pi_a = jnp.sum(actions_oh * pi, axis=-1)
    mu_a = jnp.sum(actions_oh * mu, axis=-1)
    return pi_a / mu_a


def _where(pred: chex.Array, true_data: Any, false_data: Any) -> Any:
    """jnp.where applied tree-wise, treating pred as a broadcastable prefix."""
    def _where_one(t, f):
        p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
        return jnp.where(p, t, f)
    return jax.tree.map(_where_one, true_data, false_data)


def _has_played(player_id: chex.Array, is_new_eps: chex.Array, player: int) -> chex.Array:
    """Computes a mask that is 1 at steps where `player` acts, 0 elsewhere.

    Uses a backward scan to propagate the flag correctly in the presence of
    episode boundaries (is_new_eps resets the backward carry).

    Args:
        player_id: shape (T, B) integer player ids
        is_new_eps: shape (T, B) bool, True at first step of a new episode
        player: the player index

    Returns:
        has_played: shape (T, B) float32, 1 at player's own steps, 0 elsewhere
    """
    def _loop(carry, x):
        pid, is_new = x
        result = jnp.where(pid == player, jnp.ones_like(carry), carry)
        # Reset carry going backward at episode boundaries
        carry_out = jnp.where(is_new, jnp.zeros_like(carry), carry)
        return carry_out, result

    _, result = lax.scan(
        _loop,
        jnp.zeros_like(player_id[-1], dtype=jnp.float32),
        (player_id, is_new_eps),
        reverse=True,
    )
    return result


@chex.dataclass(frozen=True)
class VTraceCarry:
    reward: chex.Array
    reward_uncorrected: chex.Array
    next_value: chex.Array
    next_v_target: chex.Array
    importance_sampling: chex.Array


def v_trace(
    v: chex.Array,
    is_new_eps: chex.Array,
    player_id: chex.Array,
    acting_policy: chex.Array,
    merged_policy: chex.Array,
    merged_log_policy: chex.Array,
    player_others: chex.Array,
    actions_oh: chex.Array,
    reward: chex.Array,
    player: int,
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
    gamma: float = 1.0,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """V-trace targets adapted for auto-reset environments.

    All shapes assume leading dimensions (T, B):
        v: (T, B, 1)
        is_new_eps: (T, B) bool
        player_id: (T, B)
        acting_policy / merged_policy / merged_log_policy: (T, B, A)
        player_others: (T, B, 1)
        actions_oh: (T, B, A)
        reward: (T, B)

    Args:
        gamma: time-discount factor. Use 1.0 (undiscounted) for finite
            zero-sum games where the full-episode return is what matters.
            Values < 1 bias toward earlier rewards, useful for long-horizon
            or continuing tasks.

    Returns:
        v_target: (T, B, 1) — v-trace value targets
        has_played: (T, B) — mask for value loss (1 at player's own steps)
        learning_output: (T, B, A) — q_vr for NERD
    """

    has_played = _has_played(player_id, is_new_eps, player)

    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh)
    inv_mu = _policy_ratio(jnp.ones_like(merged_policy), acting_policy, actions_oh)

    eta_reg_entropy = (
        -eta
        * jnp.sum(merged_policy * merged_log_policy, axis=-1)
        * jnp.squeeze(player_others, axis=-1)
    )
    eta_log_policy = -eta * merged_log_policy * player_others

    init_state = VTraceCarry(
        reward=jnp.zeros_like(reward[-1]),
        reward_uncorrected=jnp.zeros_like(reward[-1]),
        next_value=jnp.zeros_like(v[-1]),
        next_v_target=jnp.zeros_like(v[-1]),
        importance_sampling=jnp.ones_like(policy_ratio[-1]),
    )

    def _loop(carry: VTraceCarry, x) -> Tuple[VTraceCarry, Any]:
        cs, pid, v_, rew, eta_reg, is_new, inv_mu_, act_oh, eta_log = x

        reward_uncorrected = rew + gamma * carry.reward_uncorrected + eta_reg
        discounted_reward = rew + gamma * carry.reward

        our_v_target = (
            v_
            + jnp.expand_dims(jnp.minimum(rho, cs * carry.importance_sampling), axis=-1)
            * (
                jnp.expand_dims(reward_uncorrected, axis=-1)
                + gamma * carry.next_value
                - v_
            )
            + lambda_
            * jnp.expand_dims(jnp.minimum(c, cs * carry.importance_sampling), axis=-1)
            * gamma
            * (carry.next_v_target - carry.next_value)
        )
        opp_v_target = jnp.zeros_like(our_v_target)

        our_learning_output = (
            v_
            + eta_log
            + act_oh
            * jnp.expand_dims(inv_mu_, axis=-1)
            * (
                jnp.expand_dims(discounted_reward, axis=-1)
                + gamma
                * jnp.expand_dims(carry.importance_sampling, axis=-1)
                * carry.next_v_target
                - v_
            )
        )
        opp_learning_output = jnp.zeros_like(our_learning_output)

        is_our_turn = pid == player
        v_target = _where(is_our_turn, our_v_target, opp_v_target)
        learning_output = _where(is_our_turn, our_learning_output, opp_learning_output)

        our_carry = VTraceCarry(
            reward=jnp.zeros_like(carry.reward),
            reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
            next_value=v_,
            next_v_target=our_v_target,
            importance_sampling=jnp.ones_like(carry.importance_sampling),
        )
        opp_carry = VTraceCarry(
            reward=eta_reg + cs * discounted_reward,
            reward_uncorrected=reward_uncorrected,
            next_value=gamma * carry.next_value,
            next_v_target=gamma * carry.next_v_target,
            importance_sampling=cs * carry.importance_sampling,
        )

        new_carry = _where(is_our_turn, our_carry, opp_carry)
        # Reset backward carry at episode boundaries to prevent cross-episode leakage
        carry_out = _where(is_new, init_state, new_carry)

        return carry_out, (v_target, learning_output)

    _, (v_target, learning_output) = lax.scan(
        _loop,
        init_state,
        (policy_ratio, player_id, v, reward, eta_reg_entropy, is_new_eps, inv_mu, actions_oh, eta_log_policy),
        reverse=True,
    )

    return v_target, has_played, learning_output


def get_loss_v(
    v_list: list,
    v_target_list: list,
    mask_list: list,
) -> chex.Array:
    """Value loss: MSE between current value and v-trace targets, masked by has_played."""
    loss_v_list = []
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        loss_v = (
            jnp.expand_dims(mask, axis=-1) * (v_n - lax.stop_gradient(v_target)) ** 2
        )
        normalization = jnp.sum(mask)
        loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0))
        loss_v_list.append(loss_v)
    return sum(loss_v_list)


def apply_force_with_threshold(
    decision_outputs: chex.Array,
    force: chex.Array,
    threshold: float,
    threshold_center: chex.Array,
) -> chex.Array:
    """Apply force only when within threshold of threshold_center."""
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = jnp.minimum(force, 0.0)
    force_positive = jnp.maximum(force, 0.0)
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * lax.stop_gradient(clipped_force)


def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
    """Sum loss over masked positions, normalized by count."""
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def get_loss_nerd(
    logit_list: list,
    policy_list: list,
    q_vr_list: list,
    player_ids: chex.Array,
    legal_actions: chex.Array,
    importance_sampling_correction: list,
    clip: float = 100.0,
    threshold: float = 2.0,
) -> chex.Array:
    """NERD policy gradient loss.

    Args:
        logit_list: raw logits per player, each (T, B, A) — illegal actions should be 0
        policy_list: current policy probs per player, each (T, B, A)
        q_vr_list: v-trace learning outputs per player, each (T, B, A)
        player_ids: (T, B)
        legal_actions: (T, B, A) bool
        importance_sampling_correction: IS weights per player, each (T, B, 1)
        clip: advantage clipping bound
        threshold: logit threshold for force application
    """
    num_valid_actions = jnp.sum(legal_actions, axis=-1, keepdims=True)
    loss_pi_list = []

    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True)
        adv_pi = is_c * adv_pi
        adv_pi = jnp.clip(adv_pi, -clip, clip)
        adv_pi = lax.stop_gradient(adv_pi)

        # logit_pi should have 0 for illegal actions (caller's responsibility)
        valid_logit_sum = jnp.sum(logit_pi * legal_actions, axis=-1, keepdims=True)
        mean_logit = valid_logit_sum / num_valid_actions
        logits = logit_pi - mean_logit

        threshold_center = jnp.zeros_like(logits)
        nerd_loss = jnp.sum(
            legal_actions * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
            axis=-1,
        )
        player_mask = (player_ids == k).astype(jnp.float32)
        nerd_loss = -renormalize(nerd_loss, player_mask)
        loss_pi_list.append(nerd_loss)

    return sum(loss_pi_list)
