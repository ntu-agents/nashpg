import chex
import distrax

import envs.mytypes as env_types

@chex.dataclass
class Transition:
    """
    Transition data collected during trajectory rollout.
    
    Contains raw environment interactions and agent outputs.
    Shape: (num_envs, num_steps, ...)
    """
    is_new_eps: chex.Array      # bool, episode boundaries
    action: env_types.Action    # agent actions
    value: chex.Array          # critic value estimates
    reward: chex.Array         # environment rewards per agent
    log_prob: chex.Array       # action log probabilities
    observation: env_types.Observation  # environment observations
    action_mask: chex.Array    # valid action masks
    current_player: chex.Array # which agent is acting

@chex.dataclass
class RegulizedSample:
    """
    Sample data point for KL regularization
    """
    observation: env_types.Observation  # environment observations
    action_mask: chex.Array    # valid action masks
    mag_action_dist: distrax.Categorical # action distribution of magnet policy
    valid_mask: chex.Array # valid data point

@chex.dataclass
class Dataset:
    """
    Processed training dataset from transitions.
    
    Includes computed advantages and target values for training.
    Shape: (num_envs * num_steps, ...)
    """
    action: env_types.Action    # agent actions
    value: chex.Array          # critic value estimates
    log_prob: chex.Array       # action log probabilities
    observation: env_types.Observation  # environment observations
    action_mask: chex.Array    # valid action masks
    current_player: chex.Array # which agent is acting
    advantage: chex.Array      # GAE advantages
    target_value: chex.Array   # critic training targets
    valid_mask: chex.Array     # bool, valid training samples
