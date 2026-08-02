import envs.mytypes as env_types
from envs.tictactoe import TicTacToe
from envs.phantom_tictactoe import PhantomTicTacToe
from envs.dark_hex3 import DarkHex3
from envs.battleship import BattleShip
from envs.kuhn_poker import KuhnPoker
from envs.leduc_poker import LeducPoker
from envs.liar_dice import LiarDice
from envs.head_up_poker import HeadUpPoker
from enum import Enum
from typing import Union
from wrappers import AutoResetWrapper


class RegisteredEnv(Enum):
    TIC_TAC_TOE = "tic_tac_toe"
    PHANTOM_TIC_TAC_TOE_CLASSIC = "phantom_tic_tac_toe_classic"
    PHANTOM_TIC_TAC_TOE_ABRUPT = "phantom_tic_tac_toe_abrupt"
    DARK_HEX3_CLASSIC = "dark_hex3_classic"
    DARK_HEX3_ABRUPT = "dark_hex3_abrupt"
    BATTLE_SHIP = "battle_ship"
    KUHN_POKER = "kuhn_poker"
    LEDUC_POKER = "leduc_poker"
    LIAR_DICE = "liar_dice"
    HEAD_UP_POKER = "head_up_poker"


def create_env(env_name: Union[RegisteredEnv, str], auto_reset: bool = True) -> env_types.BaseEnv:
    """Create an environment instance based on the registered environment name.
    
    Args:
        env_name: The registered environment type to create (enum or string)
        auto_reset: Whether to wrap the environment with AutoResetWrapper (default: True)
        
    Returns:
        An instance of the specified environment type, optionally wrapped with AutoResetWrapper
        
    Raises:
        ValueError: If the environment name is not recognized
    """
    # Convert string to enum if needed
    if isinstance(env_name, str):
        try:
            env_name = RegisteredEnv(env_name)
        except ValueError:
            raise ValueError(f"Unknown environment: {env_name}")

    def _get_base_env(env_name: RegisteredEnv):
        if env_name == RegisteredEnv.TIC_TAC_TOE:
            return TicTacToe()
        elif env_name == RegisteredEnv.PHANTOM_TIC_TAC_TOE_CLASSIC:
            return PhantomTicTacToe(is_abrupt=False)
        elif env_name == RegisteredEnv.PHANTOM_TIC_TAC_TOE_ABRUPT:
            return PhantomTicTacToe(is_abrupt=True)
        elif env_name == RegisteredEnv.DARK_HEX3_CLASSIC:
            return DarkHex3(is_abrupt=False)
        elif env_name == RegisteredEnv.DARK_HEX3_ABRUPT:
            return DarkHex3(is_abrupt=True)
        elif env_name == RegisteredEnv.BATTLE_SHIP:
            return BattleShip()
        elif env_name == RegisteredEnv.KUHN_POKER:
            return KuhnPoker()
        elif env_name == RegisteredEnv.LIAR_DICE:
            return LiarDice()
        elif env_name == RegisteredEnv.HEAD_UP_POKER:
            return HeadUpPoker()
        elif env_name == RegisteredEnv.LEDUC_POKER:
            return LeducPoker()
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
    env = _get_base_env(env_name)
    if auto_reset:
        env = AutoResetWrapper(env)

    return env
