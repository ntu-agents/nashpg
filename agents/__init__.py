from agents.base_agent import BaseAgent
from agents.battleship import BattleShipAgent
from agents.liar_dice import LiarDiceAgent
from agents.poker import HeadUpPokerAgent
from agents.mixture_agent import MixtureAgent
from agents.random_agent import RandomAgent
from agents.mlp_agent import MLPAgent
from enum import Enum
from typing import Union
import chex


class RegisteredAgent(Enum):
    TIC_TAC_TOE = "tic_tac_toe"
    BATTLE_SHIP = "battle_ship"
    KUHN_POKER = "kuhn_poker"
    LEDUC_POKER = "leduc_poker"
    LIAR_DICE = "liar_dice"
    HEAD_UP_POKER = "head_up_poker"
    HEAD_UP_POKER_FULL = "head_up_poker_full"
    PHANTOM_TICTACTOE_CLASSIC = "phantom_tic_tac_toe_classic"
    PHANTOM_TICTACTOE_ABRUPT = "phantom_tic_tac_toe_abrupt"
    DARK_HEX3_CLASSIC = "dark_hex3_classic"
    DARK_HEX3_ABRUPT = "dark_hex3_abrupt"


def create_agent(agent_name: Union[RegisteredAgent, str], key: chex.PRNGKey) -> BaseAgent:
    if isinstance(agent_name, str):
        try:
            agent_name = RegisteredAgent(agent_name)
        except ValueError:
            raise ValueError(f"Unknown agent: {agent_name}")

    if agent_name == RegisteredAgent.TIC_TAC_TOE:
        return MLPAgent(key, input_dim=9, output_dim=9)
    elif agent_name == RegisteredAgent.BATTLE_SHIP:
        return BattleShipAgent(key)
    elif agent_name == RegisteredAgent.KUHN_POKER:
        return MLPAgent(key, input_dim=7, output_dim=2)
    elif agent_name == RegisteredAgent.LIAR_DICE:
        return LiarDiceAgent(key)
    elif agent_name == RegisteredAgent.HEAD_UP_POKER:
        return HeadUpPokerAgent(key)
    elif agent_name == RegisteredAgent.HEAD_UP_POKER_FULL:
        return HeadUpPokerAgent(key)
    elif agent_name == RegisteredAgent.LEDUC_POKER:
        return MLPAgent(key, input_dim=49, output_dim=3)
    elif agent_name == RegisteredAgent.PHANTOM_TICTACTOE_CLASSIC:
        return MLPAgent(key, input_dim=108, output_dim=9)
    elif agent_name == RegisteredAgent.PHANTOM_TICTACTOE_ABRUPT:
        return MLPAgent(key, input_dim=108, output_dim=9)
    elif agent_name == RegisteredAgent.DARK_HEX3_CLASSIC:
        return MLPAgent(key, input_dim=162, output_dim=9)
    elif agent_name == RegisteredAgent.DARK_HEX3_ABRUPT:
        return MLPAgent(key, input_dim=162, output_dim=9)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


__all__ = ['BaseAgent', 'RegisteredAgent', 'create_agent', 'MixtureAgent', 'RandomAgent']
