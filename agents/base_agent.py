from pathlib import Path
from typing import Tuple
from flax import nnx
import chex
import abc
import distrax
import orbax.checkpoint as ocp

import envs.mytypes as env_types

class BaseAgent(nnx.Module, abc.ABC):

    @abc.abstractmethod
    def __init__(self, key: chex.PRNGKey, **kwargs):
        pass

    @abc.abstractmethod
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value using the value network.
        
        Args:
            state: Observation tensor of shape (batch_size, *observation_shape)
            
        Returns:
            Value estimates of shape (batch_size,)
        """
        pass
    
    @abc.abstractmethod
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None) -> env_types.Action:
        """Sample action from policy network.
        
        Args:
            observations (batch_size, *observation_shape): Observation tensor
            key (): JAX random key for sampling
            action_masks (batch_size, num_actions): Binary mask for valid actions
            
        Returns:
            Sampled actions of shape (batch_size, )
        """
        pass
    
    @abc.abstractmethod
    def get_action_and_value(
            self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None
        ) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously.
        
        Args:
            observations (batch_size, *observation_shape): Observation tensor
            key (): JAX random key for sampling
            action_masks (batch_size, num_actions): Binary mask for valid actions
        
        Returns:
            Tuple of (action, log_prob, value) arrays of shape (batch_size,)
        """
        pass
    
    @abc.abstractmethod
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        """Get action distribution
        
        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            action_masks (batch_size, num_actions): Binary mask for valid actions
            
        Returns:
            distrax.Distribution of shape (batch_size, )
        """
        pass

    def save_checkpoint(self, checkpoint_dir: str, step: int):
        """Save agent checkpoint to disk using Orbax.
    
        Args:
            checkpoint_dir: Directory to save checkpoint
            step: step number for checkpoint naming
        """
        checkpoint_path = Path(checkpoint_dir).resolve()
        checkpoint_path = checkpoint_path / f"checkpoint_{step}"
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Split the agent into graphdef and state
        _, state = nnx.split(self)

        # Save the state using Orbax
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(checkpoint_path / 'agent_state', state)

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str, step: int, key: chex.PRNGKey, **kwargs) -> "BaseAgent":
        """Load agent checkpoint from disk using Orbax.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            step: step number for checkpoint naming
            key: JAX random key
            
        Returns:
            Loaded agent instance
        """
        checkpoint_path = Path(checkpoint_dir).resolve()
        checkpoint_path = checkpoint_path / f"checkpoint_{step}"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Create abstract model for restoration
        abstract_agent = nnx.eval_shape(lambda: cls(key, **kwargs))
        graphdef, abstract_state = nnx.split(abstract_agent)
        
        # Restore the checkpoint
        checkpointer = ocp.PyTreeCheckpointer()
        restored_state = checkpointer.restore(checkpoint_path / 'agent_state', abstract_state)
        
        # Merge graphdef and restored state to create the agent
        agent = nnx.merge(graphdef, restored_state)
        
        return agent