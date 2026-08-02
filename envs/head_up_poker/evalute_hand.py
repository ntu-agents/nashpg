"""
Poker Hand Evaluation Module

JAX-native hand evaluation. evaluate_five_cards is rewritten to use parallel
boolean indicators (instead of 9 chained jax.lax.cond calls) and a kicker table
lookup, removing all sequential data dependencies.
"""

from itertools import combinations
import chex
import jax
import jax.numpy as jnp


def get_suit(card):
    return card // 13

def get_rank(card):
    return card % 13


def evaluate_five_cards(cards):
    """
    Evaluate a five-card poker hand and return a numerical strength score.

    All hand-type indicators are computed in parallel (no sequential lax.cond
    chains). A single kicker-table lookup then assembles the final score.

    Returns a comparable integer; higher = stronger hand.
    """
    suits        = get_suit(cards)
    ranks        = get_rank(cards)
    sorted_ranks = jnp.sort(ranks)           # ascending

    # ── flush / straight ──────────────────────────────────────────────────────
    is_flush   = jnp.all(suits == suits[0])
    no_dups    = jnp.all(sorted_ranks[1:] != sorted_ranks[:-1])
    is_wheel   = jnp.all(sorted_ranks == jnp.array([0, 1, 2, 3, 12]))
    is_straight = (no_dups & (sorted_ranks[4] - sorted_ranks[0] == 4)) | is_wheel
    straight_high = jnp.where(is_wheel, jnp.int32(3), sorted_ranks[4])

    # ── rank counts ───────────────────────────────────────────────────────────
    rank_counts = jnp.bincount(ranks, length=13)     # (13,)
    max_count   = jnp.max(rank_counts)
    num_pairs   = jnp.sum(rank_counts == 2)

    # ── kicker precomputation (all computed regardless of hand type) ──────────
    four_rank  = jnp.argmax(rank_counts == 4)
    three_rank = jnp.argmax(rank_counts == 3)

    # pair ranks in descending order (two-pair needs both)
    sorted_pair_ranks = jnp.sort(jnp.where(rank_counts == 2, jnp.arange(13), -1))[::-1]
    high_pair = sorted_pair_ranks[0]
    low_pair  = sorted_pair_ranks[1]

    kicker_4ok = jnp.max(jnp.where(ranks != four_rank,   ranks, -1))
    kicker_fh  = jnp.argmax(rank_counts == 2)             # pair rank in full house
    kicker_2p  = jnp.max(jnp.where((ranks != high_pair) & (ranks != low_pair), ranks, -1))

    three_rest = jnp.sort(jnp.where(ranks != three_rank, ranks, -1))[::-1]
    k1_3ok = three_rest[0]
    k2_3ok = three_rest[1]

    op_rank    = jnp.argmax(rank_counts == 2)
    op_rest    = jnp.sort(jnp.where(ranks != op_rank, ranks, -1))[::-1]
    k1_1p = op_rest[0]
    k2_1p = op_rest[1]
    k3_1p = op_rest[2]

    # ── parallel hand-type indicators (all are mutually exclusive) ────────────
    # (verified: flush/straight hands cannot overlap with pair/trips hands
    #  because equal-rank cards require different suits in a standard 52-card deck)
    is_rf  = is_flush & is_straight & (straight_high == 12)
    is_sf  = is_flush & is_straight & ~is_rf
    is_4ok = max_count == 4
    is_fh  = (max_count == 3) & (num_pairs >= 1)
    is_fl  = is_flush & ~is_straight
    is_st  = is_straight & ~is_flush
    is_3ok = (max_count == 3) & (num_pairs == 0)
    is_2p  = num_pairs == 2
    is_1p  = (num_pairs == 1) & (max_count < 3)
    # high card: all above are False → hand_type = 0

    hand_type = (
        9 * is_rf  + 8 * is_sf  + 7 * is_4ok + 6 * is_fh +
        5 * is_fl  + 4 * is_st  + 3 * is_3ok + 2 * is_2p + 1 * is_1p
    ).astype(jnp.int32)

    # ── kicker table: shape (10, 5), indexed by hand_type ─────────────────────
    # Each row: [kicker0, kicker1, kicker2, kicker3, kicker4]  (unused → 0)
    # Score = hand_type * 13^5  +  dot(kicker_row, [13^4, 13^3, 13^2, 13, 1])
    Z = jnp.array(0, dtype=jnp.int32)

    kicker_table = jnp.stack([
        # 0 High Card:   5 ranks descending
        jnp.stack([sorted_ranks[4], sorted_ranks[3], sorted_ranks[2],
                   sorted_ranks[1], sorted_ranks[0]]),
        # 1 One Pair:    pair rank, then 3 kickers
        jnp.stack([op_rank, k1_1p, k2_1p, k3_1p, Z]),
        # 2 Two Pair:    high pair, low pair, kicker
        jnp.stack([high_pair, low_pair, kicker_2p, Z, Z]),
        # 3 Three of a Kind: three rank, 2 kickers
        jnp.stack([three_rank, k1_3ok, k2_3ok, Z, Z]),
        # 4 Straight:    straight-high card
        jnp.stack([straight_high, Z, Z, Z, Z]),
        # 5 Flush:       5 ranks descending (same as HC row but under flush type)
        jnp.stack([sorted_ranks[4], sorted_ranks[3], sorted_ranks[2],
                   sorted_ranks[1], sorted_ranks[0]]),
        # 6 Full House:  three rank, pair rank
        jnp.stack([three_rank, kicker_fh, Z, Z, Z]),
        # 7 Four of a Kind: four rank, kicker
        jnp.stack([four_rank, kicker_4ok, Z, Z, Z]),
        # 8 Straight Flush: straight-high
        jnp.stack([straight_high, Z, Z, Z, Z]),
        # 9 Royal Flush: ace high (always 12)
        jnp.stack([jnp.array(12, dtype=jnp.int32), Z, Z, Z, Z]),
    ])  # (10, 5)

    W = jnp.array([13**4, 13**3, 13**2, 13**1, 13**0], dtype=jnp.int32)
    kicker_vec = kicker_table[hand_type]          # (5,)
    score = hand_type * (13**5) + jnp.dot(kicker_vec, W)
    return score


def evaluate_hand(hole_cards: chex.Array, community_cards: chex.Array) -> chex.Array:
    """
    Evaluate the best possible 5-card hand from 7 available cards.

    Generates all 21 five-card combinations and returns the maximum score.
    """
    comb_indices = jnp.array(list(combinations(range(7), 5)))   # (21, 5)
    all_cards    = jnp.concatenate([hole_cards, community_cards])  # (7,)
    selected     = all_cards[comb_indices]                          # (21, 5)
    scores       = jax.vmap(evaluate_five_cards)(selected)          # (21,)
    return jnp.max(scores)
