"""
No Limit Head-Up Texas Hold'em Environment for Multi-Agent Reinforcement Learning

no-limit Texas Hold'em poker environment optimized for two-player (heads-up) games.
The environment follows standard poker rules.

Card Encoding:
    Cards are encoded as integers from 0-51:
    - Suits: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades  
    - Ranks: 0=2, 1=3, ..., 11=King, 12=Ace
    - Card value = rank + 13 * suit

Environment Configuration:
    - Starting stack: 200 chips per player
    - Small blind: 1 chip, Big blind: 2 chips
    - Maximum action history: `MAX_HISTORY_LENGTH` actions per hand
    - 16 possible actions per turn (fold, check/call, 12 raise sizes, all-in)
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

from envs.head_up_poker.evalute_hand import evaluate_hand
from envs.mytypes import TimeStep, BaseEnv, Action
import envs.myspaces as env_spaces


# --- Environment Constants ---
STARTING_STACK = 200
SMALL_BLIND = 1
BIG_BLIND = 2
MAX_HISTORY_LENGTH = 64

# --- Action Constants ---
# Action encoding: 0=Fold, 1=Check, 2=Call, 3-14=Raise Fractions, 15=All-in
NUM_ACTIONS = 16  # Total number of possible actions
RAISE_FRACTIONS = jnp.array([1/4, 1/3, 1/2, 2/3, 3/4, 1.0, 4/3, 3/2, 7/4, 2.0, 3.0, 4.0])  # Pot fraction multipliers for raises
FOLD, CHECK, CALL = 0, 1, 2  # Basic actions
RAISE_START = 3  # First raise action index
ALL_IN = 15  # All-in action index

# --- Street Constants ---
# Betting round progression through Texas Hold'em streets
PREFLOP, FLOP, TURN, RIVER, SHOWDOWN = 0, 1, 2, 3, 4


@chex.dataclass
class EnvState:
    """
    Complete state representation for the poker environment.
    
    Contains all information needed to represent the current game state,
    including player stacks, cards, betting state, and action history.
    All fields are JAX arrays for efficient computation.
    """
    key: chex.PRNGKey  # JAX random key for stochastic operations
    
    # --- Global Game State ---
    done: chex.Numeric  # Boolean: True if match is over (player bankrupt)
    step_cnt: chex.Numeric  # Total steps taken in the match
    
    # --- Hand-specific State ---
    deck: chex.Array  # (52,) Shuffled deck of cards [0-51]
    dealer: chex.Numeric  # Dealer button position (0 or 1)
    stacks: chex.Array  # (2,) Current chip counts for each player
    initial_stacks: chex.Array  # (2,) Stacks at start of hand (for reward calculation)
    hole_cards: chex.Array  # (2, 2) Private cards for each player
    board_cards: chex.Array  # (5,) Community cards, -1 if not yet dealt
    street: chex.Numeric  # Current betting round: PREFLOP=0, FLOP=1, TURN=2, RIVER=3, SHOWDOWN=4
    pot_size: chex.Numeric  # Total chips in the main pot
    bets: chex.Array  # (2,) Chips committed by each player in current street
    committed_this_hand: chex.Array  # (2,) Total chips committed by each player this hand
    min_raise_amount: chex.Numeric  # Minimum raise amount (by this much)
    raise_is_closed: chex.Numeric  # Boolean: True if incomplete raise closes action
    actions_this_street: chex.Numeric  # Number of actions taken in current betting round
    
    # --- Player Action State ---
    current_player: chex.Numeric  # Player to act next (0 or 1)
    aggressor: chex.Numeric  # Last player to bet/raise, -1 if none this street
    player_folded: chex.Array  # (2,) Boolean flags for folded players
    player_all_in: chex.Array  # (2,) Boolean flags for all-in players
    
    # --- Action History ---
    action_history: chex.Array  # (MAX_HISTORY_LENGTH, 4) Action log: [player, street, action, amount]
    history_idx: chex.Numeric  # Current index in action_history buffer


class HeadUpPoker(BaseEnv):
    """
    JAX-native No-Limit Texas Hold'em Heads-Up Environment.
    
    This environment implements a complete two-player poker game with proper betting
    mechanics, hand evaluation, and pot distribution. All operations are JAX-compiled
    for high performance training.
    
    Key Methods:
        reset(key): Initialize a new game with random dealer
        step(state, action): Execute one action and return new state/timestep
        
    State Management:
        - Game progresses through streets: PREFLOP -> FLOP -> TURN -> RIVER -> SHOWDOWN
        - Each hand starts with random dealer, blinds posted automatically
        - Players alternate acting until betting round complete
        - Pot settled at showdown or when player folds
    """
    def __init__(self):
        super().__init__()

    @property
    def env_name(self) -> str:
        return "no_limit_poker"

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def action_space(self) -> env_spaces.Discrete:
        """
        Action space with 16 discrete actions:
        
        Actions:
            0: Fold - Forfeit hand and lose committed chips
            1: Check - Pass action (only when no bet to call)
            2: Call - Match opponent's bet
            3-14: Raise - Fractional pot-sized raises (0.25x to 4x pot)
            15: All-in - Bet entire remaining stack
            
        Returns:
            Discrete space with NUM_ACTIONS=16 categories
        """
        return env_spaces.Discrete(num_categories=NUM_ACTIONS)

    @property
    def observation_space(self) -> env_spaces.Dict:
        """
        Observation space containing all visible game information from current player's perspective.
        
        Observation Fields:
            hole_cards: (2,) Player's private cards [0-51]
            board_cards: (5,) Community cards [-1 if not dealt, 0-51 if dealt]
            pot_size: () Total chips in pot [0 to 2*STARTING_STACK]
            street: () Current betting round [0=PREFLOP, 1=FLOP, 2=TURN, 3=RIVER]
            stacks: (2,) Chip counts [current_player, opponent] 
            bets: (2,) Current street bets [current_player, opponent]
            action_history: (MAX_HISTORY_LENGTH, 4) Action log [player_id, street, action, amount]
            bet_amount: (NUM_ACTIONS,) The chip amount for each corresponding action
            
        Note: stacks, bets, and action_history are from current player's POV. (0=self, 1=opponent)
        
        Returns:
            Dict space with all observation components
        """
        return env_spaces.Dict({
            "hole_cards": env_spaces.Box(low=0, high=51, shape=(2,), dtype=jnp.int32),
            "board_cards": env_spaces.Box(low=-1, high=51, shape=(5,), dtype=jnp.int32),
            "pot_size": env_spaces.Box(low=0, high=STARTING_STACK * 2, shape=(), dtype=jnp.int32),
            "street": env_spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "stacks": env_spaces.Box(low=0, high=STARTING_STACK * 2, shape=(2,), dtype=jnp.int32),
            "bets": env_spaces.Box(low=0, high=STARTING_STACK * 2, shape=(2,), dtype=jnp.int32),
            "action_history": env_spaces.Box(low=-1, high=STARTING_STACK * 2, shape=(MAX_HISTORY_LENGTH, 4), dtype=jnp.int32), # player POV (0=current_player, 1=opponent)
            "bet_amount": env_spaces.Box(low=0, high=STARTING_STACK * 2, shape=(NUM_ACTIONS,), dtype=jnp.int32)
        })

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        """
        Resets the environment to start a new poker match.
        
        Initializes both players with STARTING_STACK chips, randomly selects
        the initial dealer, and deals the first hand. The dealer button will
        alternate between players for subsequent hands.
        
        Args:
            key: JAX random key for shuffling and dealer selection
            
        Returns:
            state: Initial environment state with first hand dealt
            timestep: Initial observation for the first player to act
        """

        initial_stacks = jnp.array([STARTING_STACK, STARTING_STACK], dtype=jnp.int32)

        # A random player starts as dealer
        key, subkey = jax.random.split(key)
        start_dealer = jax.random.bernoulli(key=subkey, p=0.5).astype(jnp.int32)

        # Initial state before the first hand is dealt
        state = EnvState(
            key=key,
            done=jnp.bool_(False),
            step_cnt=jnp.int32(0),
            deck=jnp.arange(52, dtype=jnp.int32),
            dealer=start_dealer,
            stacks=initial_stacks,
            initial_stacks=initial_stacks,
            hole_cards=jnp.full((2, 2), -1, dtype=jnp.int32),
            board_cards=jnp.full((5,), -1, dtype=jnp.int32),
            street=jnp.int32(PREFLOP),
            pot_size=jnp.int32(0),
            bets=jnp.zeros(2, dtype=jnp.int32),
            committed_this_hand=jnp.zeros(2, dtype=jnp.int32),
            min_raise_amount=jnp.int32(0),
            raise_is_closed=jnp.bool_(0),
            actions_this_street=jnp.int32(0),
            current_player=start_dealer,
            aggressor=jnp.int32(-1),
            player_folded=jnp.zeros(2, dtype=jnp.bool_),
            player_all_in=jnp.zeros(2, dtype=jnp.bool_),
            action_history=jnp.full((MAX_HISTORY_LENGTH, 4), -1, dtype=jnp.int32),
            history_idx=jnp.int32(0)
        )

        state = self._deal_new_hand(state)
        return state, self._get_timestep(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        """
        Executes one action in the poker environment.
        
        Processes the current player's action, updates game state, and determines
        next player or street progression. Handles hand completion, pot settlement,
        and automatic dealing of new hands.
        
        Args:
            state: Current environment state
            action: Integer action [0-11] from current player
            
        Returns:
            state: Updated environment state
            timestep: Observation for next player with rewards (if hand completed)
        """

        # If the game is already done, do nothing.
        state, ts = jax.lax.cond(
            state.done,
            lambda s, _: (s, self._get_timestep(s)),
            self._step_fn,
            state, action
        )
        return state, ts

    def _step_fn(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        """The core logic of a single step."""
        # 1. Apply the action taken by the current player
        state = self._apply_action(state, action)

        # 2. Check if the betting round is over
        is_round_over = self._is_round_over(state)

        # 3. If round is over, proceed to next street or showdown. Otherwise, switch player.
        state = jax.lax.cond(
            is_round_over,
            self._end_betting_round,
            self._continue_betting_round,
            state
        )

        # 4. Check if the hand has concluded (due to fold, showdown, etc.)
        is_hand_over = self._is_hand_over(state)

        # 5. If hand is over, settle the pot and deal a new hand.
        #    The function returns the new state and the reward from the concluded hand.
        state, reward = jax.lax.cond(
            is_hand_over,
            self._end_hand,
            lambda s: (s, jnp.zeros(2, dtype=jnp.float32)), # No reward if hand continues
            state
        )

        # 6. Check if the entire game is over (a player is bankrupt)
        done = is_hand_over & jnp.any(state.stacks <= 0)
        state = state.replace(done=done)

        # 7. Construct and return the timestep
        ts = self._get_timestep(state, reward, is_hand_over)
        return state.replace(step_cnt=state.step_cnt + 1), ts

    def _deal_new_hand(self, state: EnvState) -> EnvState:
        """
        Deals a new poker hand with shuffled deck and posted blinds.
        
        Shuffles deck, deals 2 hole cards to each player, posts blinds according
        to dealer position, and sets up initial betting state. In heads-up poker,
        the dealer is the small blind and acts first preflop.
        
        Args:
            state: Current environment state
            
        Returns:
            Updated state with new hand ready for first action
        """
        key, subkey = jax.random.split(state.key, 2)

        # Capture the stacks at the beginning of the hand
        hand_initial_stacks = state.stacks

        # Shuffle deck
        deck = jax.random.permutation(subkey, jnp.arange(52, dtype=jnp.int32))

        # Deal hole cards
        hole_cards = jnp.array([[deck[0], deck[2]], [deck[1], deck[3]]], dtype=jnp.int32)

        # Post blinds
        dealer = state.dealer
        small_blinder = dealer
        big_blinder = 1 - dealer

        stacks = state.stacks
        sb_amount = jnp.minimum(SMALL_BLIND, stacks[small_blinder])
        bb_amount = jnp.minimum(BIG_BLIND, stacks[big_blinder])

        bets = jnp.zeros(2, dtype=jnp.int32).at[small_blinder].set(sb_amount)
        bets = bets.at[big_blinder].set(bb_amount)

        stacks = stacks.at[small_blinder].add(-sb_amount)
        stacks = stacks.at[big_blinder].add(-bb_amount)

        pot_size = sb_amount + bb_amount
        
        committed_this_hand = bets

        # Pre-flop, player after big blind acts first. In HU, this is the SB/dealer.
        current_player = small_blinder

        return state.replace(
            key=key,
            deck=deck,
            stacks=stacks,
            initial_stacks=hand_initial_stacks, # Store stacks for reward calculation
            hole_cards=hole_cards,
            board_cards=jnp.full((5,), -1, dtype=jnp.int32),
            street=jnp.int32(PREFLOP),
            pot_size=pot_size,
            bets=bets,
            committed_this_hand=committed_this_hand,
            current_player=current_player,
            aggressor=big_blinder,
            player_folded=jnp.zeros(2, dtype=jnp.bool_),
            player_all_in=(stacks <= 0),
            actions_this_street=jnp.int32(0),
            min_raise_amount=jnp.int32(BIG_BLIND),
            raise_is_closed=jnp.bool_(0),
            action_history=jnp.full((MAX_HISTORY_LENGTH, 4), -1, dtype=jnp.int32),
            history_idx=jnp.int32(0)
        )

    def _apply_action(self, state: EnvState, action: Action) -> EnvState:
        """
        Processes a player's action and updates the game state.
        
        Calculates bet amounts for raises based on pot size and fraction,
        updates player stacks and pot, tracks betting history, and manages
        raise/re-raise mechanics including minimum raise requirements.
        
        Args:
            state: Current environment state
            action: Player's chosen action [0-11]
            
        Returns:
            Updated state with action effects applied
        """
        p = state.current_player
        o = 1 - p

        # Calculate raise amount based on action
        amount_to_call = state.bets[o] - state.bets[p]
        # The pot for a "pot-sized raise" includes the current pot + the call amount.
        effective_pot_for_raise = state.pot_size + amount_to_call
        
        def _calculate_bet_amount(act):
            def _fold(): return 0
            def _check(): return 0
            def _call(): return amount_to_call
            def _all_in(): return state.stacks[p]
            def _raise():
                # The raise is BY this amount, on top of the call
                raise_by_amount = (RAISE_FRACTIONS[act - RAISE_START] * effective_pot_for_raise).astype(jnp.int32)
                return amount_to_call + raise_by_amount

            return jax.lax.switch(
                act,
                [_fold, _check, _call] + ([_raise] * len(RAISE_FRACTIONS)) + [_all_in]
            )
        
        bet_amount = _calculate_bet_amount(action)

        # Ensure bet does not exceed stack
        bet_amount = jnp.minimum(bet_amount, state.stacks[p])

        # Record action to history with wrap-around
        history_entry = jnp.array([
            p,                      # Player who acted (absolute index)
            state.street,           # Current street
            action,                 # Action taken
            bet_amount              # Amount committed in this action
        ], dtype=jnp.int32)
        
        new_action_history = state.action_history.at[state.history_idx].set(history_entry)
        new_history_idx = (state.history_idx + 1) % MAX_HISTORY_LENGTH

        # Update state based on action
        is_fold = (action == FOLD)
        is_all_in = (bet_amount == state.stacks[p])

        new_stacks = state.stacks.at[p].add(-bet_amount)
        new_bets = state.bets.at[p].add(bet_amount)
        new_pot = state.pot_size + bet_amount
        
        # Update total committed chips for the hand
        new_committed_this_hand = state.committed_this_hand.at[p].add(bet_amount)

        is_actual_raise = new_bets[p] > state.bets[o]
        raise_delta = new_bets[p] - state.bets[o]
        
        # The minimum amount a valid raise must be *by*. If no raise yet this street, it's the Big Blind.
        min_raise_by_amount = jnp.where(state.min_raise_amount > 0, state.min_raise_amount, jnp.int32(BIG_BLIND))
        
        is_full_raise = is_actual_raise & (raise_delta >= min_raise_by_amount)
        is_incomplete_raise = is_actual_raise & ~is_full_raise

        # A full raise opens the action. An incomplete raise closes it.
        # A call or check does not change the status.
        new_raise_is_closed = jnp.where(is_full_raise,
                                        jnp.bool_(0),
                                        jnp.where(is_incomplete_raise, jnp.bool_(1), state.raise_is_closed)
                                    )

        # Update the minimum raise amount only if a full raise was made.
        new_min_raise_amount = jnp.where(is_full_raise, raise_delta, state.min_raise_amount)
        
        new_aggressor = jnp.where(is_actual_raise, p, state.aggressor)

        return state.replace(
            stacks=new_stacks,
            bets=new_bets,
            pot_size=new_pot,
            committed_this_hand=new_committed_this_hand,
            player_folded=state.player_folded.at[p].set(is_fold),
            player_all_in=state.player_all_in.at[p].set(is_all_in),
            aggressor=new_aggressor,
            min_raise_amount=new_min_raise_amount,
            raise_is_closed=new_raise_is_closed,
            actions_this_street=state.actions_this_street + 1,
            action_history=new_action_history,
            history_idx=new_history_idx
        )

    def _is_round_over(self, state: EnvState) -> chex.Numeric:
        """Checks if the current betting round has concluded."""
        # Round always ends if a player folds or not all players can act.
        player_cannot_act = state.player_folded | state.player_all_in
        folded = jnp.any(state.player_folded)
        # betting is capped if (1) any player fold. (2) our opponent can't act
        betting_capped = folded | player_cannot_act[1 - state.current_player]

        bets_equal = (state.bets[0] == state.bets[1])
        
        # Have players had a chance to act?
        actions_taken = state.actions_this_street >= 2
        
        # Betting is complete if bets are equal and players have acted
        betting_complete = bets_equal & actions_taken
        
        return betting_capped | betting_complete

    def _continue_betting_round(self, state: EnvState) -> EnvState:
        """Switches to the next player in the current betting round."""
        return state.replace(current_player=1 - state.current_player)

    def _end_betting_round(self, state: EnvState) -> EnvState:
        """
        Advances to the next street or showdown.
        Pot settlement and refunds are handled exclusively in _end_hand.
        """
        next_street = jnp.where(
            jnp.any(state.player_folded) | jnp.any(state.player_all_in),
            SHOWDOWN,
             state.street + 1
        )

        new_board_cards = jnp.where(next_street > jnp.array([PREFLOP, PREFLOP, PREFLOP, FLOP, TURN], jnp.int32),
            jnp.array([state.deck[5], state.deck[6], state.deck[7], state.deck[9], state.deck[11]], dtype=jnp.int32),
            jnp.full((5,), -1, dtype=jnp.int32),
        )

        return state.replace(
            street=next_street,
            board_cards=new_board_cards,
            bets=jnp.zeros(2, dtype=jnp.int32), # Reset bets for the new street
            current_player=state.dealer, # The dealer (Small Blind in HU) acts first.
            aggressor=-1,
            actions_this_street=jnp.int32(0),
            min_raise_amount=jnp.int32(BIG_BLIND),
            raise_is_closed=jnp.bool_(0)
        )

    def _is_hand_over(self, state: EnvState) -> chex.Numeric:
        """Checks if the entire hand is over."""
        return state.street == SHOWDOWN

    def _end_hand(self, state: EnvState) -> Tuple[EnvState, chex.Array]:
        """
        Settles the pot, calculates rewards, and prepares for the next hand.
        This is the single source of truth for pot distribution.
        """
        # --- 1. Determine Winner ---
        p0_folded = state.player_folded[0]
        p1_folded = state.player_folded[1]

        # Skip hand evaluation when someone folded — fold flags override the winner.
        someone_folded = p0_folded | p1_folded
        dummy = jnp.int32(0)
        p0_score = jax.lax.cond(
            someone_folded,
            lambda _: dummy,
            lambda _: evaluate_hand(state.hole_cards[0], state.board_cards),
            None,
        )
        p1_score = jax.lax.cond(
            someone_folded,
            lambda _: dummy,
            lambda _: evaluate_hand(state.hole_cards[1], state.board_cards),
            None,
        )

        winner = jnp.where(p0_score > p1_score, 0,
                           jnp.where(p1_score > p0_score, 1, -1))
        # A fold overrides the showdown result.
        winner = jnp.where(p0_folded, 1, winner)
        winner = jnp.where(p1_folded, 0, winner)

        # --- 2. Settle the Pot ---
        p0_committed = state.committed_this_hand[0]
        p1_committed = state.committed_this_hand[1]

        # The amount each player contests is the minimum of their commitments.
        # This forms the main pot.
        min_committed = jnp.minimum(p0_committed, p1_committed)
        main_pot_size = min_committed * 2

        # Any amount committed beyond the minimum is an uncalled bet and must be refunded.
        # This is effectively the "side pot" in a heads-up match.
        refund_p0 = p0_committed - min_committed
        refund_p1 = p1_committed - min_committed

        # Start with the stacks as they were at the end of betting.
        new_stacks = state.stacks
        # Add the refunds back to the players' stacks.
        new_stacks = new_stacks.at[0].add(refund_p0)
        new_stacks = new_stacks.at[1].add(refund_p1)

        # Handle a tie: split the main pot.
        # The odd chip (if any) goes to the player in the worst position (the dealer/SB).
        def split_pot(s):
            pot_half = main_pot_size // 2
            odd_chip = main_pot_size % 2
            s = s.at[:].add(pot_half) # Each player gets half
            s = s.at[state.dealer].add(odd_chip) # Dealer gets the odd chip
            return s
        
        # Distribute the main pot based on the winner (or tie).
        new_stacks = jax.lax.cond(winner == -1,
            split_pot,
            lambda s: s.at[winner].add(main_pot_size),
            new_stacks
        )

        # --- 3. Calculate Rewards & Prepare Next Hand ---
        # Reward is the change in stack size from the beginning of the hand, normalized by starting stack.
        reward = (new_stacks - state.initial_stacks) / float(STARTING_STACK)

        # Check for game over before dealing the next hand.
        done = jnp.any(new_stacks <= 0)

        # Prepare for next hand
        next_hand_dealer = 1 - state.dealer
        # Create the state for the next hand by updating stacks and dealer.
        next_state = state.replace(stacks=new_stacks, dealer=next_hand_dealer)

        # Deal a new hand, unless the game is over.
        # This prevents unnecessary work on the final step.
        next_state = jax.lax.cond(
            done,
            lambda s: s, # If game over, return state as is.
            self._deal_new_hand,
            next_state
        )

        return next_state, reward

    def _get_action_mask(self, state: EnvState) -> chex.Array:
        """Generates boolean mask indicating which actions are legal."""
        p = state.current_player
        o = 1 - p

        amount_to_call      = state.bets[o] - state.bets[p]
        effective_pot       = state.pot_size + amount_to_call
        can_legally_raise   = ~state.raise_is_closed.astype(jnp.bool_)
        min_raise_by_amount = jnp.where(
            state.min_raise_amount > 0, state.min_raise_amount, jnp.int32(BIG_BLIND)
        )

        # Vectorized raise mask: all 12 raise sizes evaluated in one broadcast
        raise_by_amounts = (RAISE_FRACTIONS * effective_pot).astype(jnp.int32)
        legal_raise_size = raise_by_amounts >= min_raise_by_amount
        can_afford_raise = state.stacks[p] >= amount_to_call + raise_by_amounts
        raise_mask = can_legally_raise & legal_raise_size & can_afford_raise

        return jnp.concatenate([
            jnp.array([
                state.bets[p] < state.bets[o],  # FOLD: only valid when facing a bet
                state.bets[p] >= state.bets[o], # CHECK
                amount_to_call > 0,             # CALL
            ], dtype=jnp.bool_),
            raise_mask,
            jnp.array([can_legally_raise], dtype=jnp.bool_),  # ALL_IN
        ])
    
    def _get_bet_amounts(self, state: EnvState) -> chex.Array:
        """
        Calculates the absolute chip amount for each possible action to be included
        in the observation.
        
        Args:
            state: Current environment state
            
        Returns:
            An array of shape (NUM_ACTIONS,) with the chip amount for each action.
        """
        p = state.current_player
        o = 1 - p
        player_stack = state.stacks[p]
        
        amount_to_call = state.bets[o] - state.bets[p]
        effective_pot_for_raise = state.pot_size + amount_to_call

        # Action 0, 1: Fold, Check
        fold_amount = 0
        check_amount = 0
        
        # Action 2: Call
        call_amount = amount_to_call
        
        # Actions 3-10: Raises
        raise_by_amounts = (RAISE_FRACTIONS * effective_pot_for_raise).astype(jnp.int32)
        raise_total_bets = amount_to_call + raise_by_amounts
        
        # Action 11: All-in
        all_in_amount = player_stack

        # Assemble the array
        bet_amounts = jnp.zeros(NUM_ACTIONS, dtype=jnp.int32)
        bet_amounts = bet_amounts.at[FOLD].set(fold_amount)
        bet_amounts = bet_amounts.at[CHECK].set(check_amount)
        bet_amounts = bet_amounts.at[CALL].set(call_amount)
        bet_amounts = bet_amounts.at[RAISE_START:ALL_IN].set(raise_total_bets)
        bet_amounts = bet_amounts.at[ALL_IN].set(all_in_amount)

        # Ensure amounts are non-negative and do not exceed the player's stack
        bet_amounts = jnp.maximum(0, bet_amounts)
        bet_amounts = jnp.minimum(bet_amounts, player_stack)
        
        return bet_amounts

    def _get_timestep(self, state: EnvState, reward: chex.Array = None, hand_done: chex.Numeric = None) -> TimeStep:
        """
        Constructs observation timestep from current player's perspective.
        
        Transforms game state into player-centric observation with current player
        always at index 0. Includes action mask for legal moves and reward signal
        when hands complete.
        
        Args:
            state: Current environment state
            reward: Optional reward array [player0_reward, player1_reward]
            hand_done: Optional flag indicating if hand just completed
            
        Returns:
            TimeStep with observation, reward, done flag, and action mask
        """
        if reward is None:
            reward = jnp.zeros(2, dtype=jnp.float32)
        if hand_done is None:
            hand_done = jnp.bool_(False)

        p = state.current_player

        # Convert Action History to Player's POV
        history = state.action_history
        recorded_players = history[:, 0]
        
        # Create a new player column for the observation.
        pov_player_col = jnp.where(recorded_players == p, 0, 1)
        pov_player_col = jnp.where(recorded_players == -1, -1, pov_player_col)
        
        pov_action_history = history.at[:, 0].set(pov_player_col)

        obs = {
            "hole_cards": state.hole_cards[p],
            "board_cards": state.board_cards,
            "pot_size": state.pot_size,
            "street": state.street,
            "stacks": jnp.roll(state.stacks, -p),
            "bets": jnp.roll(state.bets, -p),
            "action_history": pov_action_history,
            "bet_amount": self._get_bet_amounts(state),
        }

        return TimeStep(
            reward=reward,
            done=state.done,
            observation=obs,
            action_mask=self._get_action_mask(state),
            current_player=p,
            info={'hand_done': hand_done, "step_cnt": state.step_cnt}
        )