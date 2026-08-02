import abc
import chex
from typing import Tuple, Union, Dict

from envs.myspaces import Space

Observation = Union[chex.Array, Dict[str, chex.Array]]
Action = Union[chex.Array, Dict[str, chex.Array]]

@chex.dataclass
class TimeStep:
    reward: chex.Array # (num_agents, )
    done: chex.Numeric # ()
    observation: Observation # (*obs_shape, )
    action_mask: Action # (*obs_shape, )
    current_player: chex.Numeric # ()
    info: Dict[str, chex.Array]


@chex.dataclass
class EnvState:
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric # when done == False, step should raise error
    step_cnt: chex.Numeric

class BaseEnv(abc.ABC):
    """
    
    Assumption:
        1) agents share same action space and observation space
        2) all agent terminate at same step
    """

    @property
    @abc.abstractmethod
    def env_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def num_agents(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def action_space(self) -> Space:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self) -> Space:
        pass

    @abc.abstractmethod
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        pass

    @abc.abstractmethod
    def step(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        pass

    

