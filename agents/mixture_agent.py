from typing import List, Tuple
from flax import nnx
import chex
import distrax
import jax.numpy as jnp
import jax

import envs.mytypes as env_types
from agents import BaseAgent


class MixtureWeights(nnx.Variable):
    """Non-trainable variable for mixture probabilities/logits.

    Using a dedicated nnx.Variable subclass makes NNX treat these as dynamic
    inputs to nnx.jit. Updating .value between outer iterations does NOT
    trigger recompilation.
    """
    pass


class MixtureAgent(BaseAgent):
    """JAX-compatible mixture agent with a pre-allocated fixed-capacity pool.

    Pre-allocate to full capacity upfront so array shapes never change and
    nnx.jit compiles exactly once. Inactive slots hold logit = -inf so
    jax.random.categorical never samples them. Fill slots progressively via
    update_slot + update_mixture_probs.

    Preferred construction:
        meta = MixtureAgent.create_with_capacity(key, template_agent, capacity=N)
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        agents: List[BaseAgent],
        mixture_logits: chex.Array = None,
        mixture_probs: chex.Array = None,
    ):
        assert agents is not None and len(agents) > 0
        if mixture_logits is not None and mixture_probs is not None:
            raise ValueError("Cannot specify both mixture_logits and mixture_probs")
        if mixture_logits is None and mixture_probs is None:
            raise ValueError("Must specify either mixture_logits or mixture_probs")

        if mixture_logits is not None:
            assert len(agents) == len(mixture_logits)
            logits = mixture_logits
            probs = jax.nn.softmax(logits)
        else:
            assert len(agents) == len(mixture_probs)
            probs = mixture_probs
            logits = jnp.where(probs > 0, jnp.log(probs), jnp.full_like(probs, -jnp.inf))

        self.mixture_logits = MixtureWeights(logits)
        self.mixture_probs = MixtureWeights(probs)
        self.num_agents = len(agents)
        self.agents = agents

        graphdef, _ = nnx.split(agents[0])
        self.agent_graphdef = graphdef

    @classmethod
    def create_with_capacity(
        cls,
        key: chex.PRNGKey,
        template_agent: BaseAgent,
        capacity: int,
    ) -> "MixtureAgent":
        """Build a MixtureAgent pre-allocated to capacity slots.

        Slot 0 is active (logit=0). Slots 1..capacity-1 are dummy (logit=-inf),
        never sampled until activated. Constant shapes → nnx.jit compiles once.
        """
        agents = [nnx.clone(template_agent) for _ in range(capacity)]
        logits = jnp.full(capacity, -jnp.inf, dtype=jnp.float32).at[0].set(0.0)
        return cls(key, agents=agents, mixture_logits=logits)

    def update_slot(self, slot_idx: int, new_agent: BaseAgent) -> None:
        """Copy new_agent's parameters into pre-allocated slot slot_idx in-place."""
        nnx.update(self.agents[slot_idx], nnx.state(new_agent))

    def update_mixture_probs(self, new_probs: jnp.ndarray) -> None:
        """Update mixture probabilities in-place. new_probs must have length == capacity."""
        self.mixture_probs.value = new_probs
        self.mixture_logits.value = jnp.where(
            new_probs > 0, jnp.log(new_probs), jnp.full_like(new_probs, -jnp.inf)
        )

    def _reconstruct_agent(self, agent_idx: chex.Numeric) -> BaseAgent:
        if isinstance(agent_idx, int):
            return self.agents[agent_idx]
        all_states = [nnx.split(a)[1] for a in self.agents]
        stacked = jax.tree.map(lambda *s: jnp.stack(s, axis=0), *all_states)
        selected = jax.tree.map(lambda x: x[agent_idx], stacked)
        return nnx.merge(self.agent_graphdef, selected)

    def get_value(self, observations: env_types.Observation) -> chex.Array:
        return self.get_value_by_index(observations, 0)

    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None) -> env_types.Action:
        return self.get_action_and_value_by_index(observations, key, action_masks, 0)

    def get_action_and_value(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        return self.get_action_and_value_by_index(observations, key, action_masks, 0)

    def get_action_distribution(self, observations: env_types.Observation, action_masks: chex.Array = None) -> distrax.Distribution:
        return self.get_action_distribution_by_index(observations, action_masks, 0)

    def get_action_distribution_by_index(self, observations: env_types.Observation, action_masks: chex.Array, agent_idx: int) -> distrax.Distribution:
        return self._reconstruct_agent(agent_idx).get_action_distribution(observations, action_masks)

    def get_action_and_value_by_index(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array, agent_idx: int) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        return self._reconstruct_agent(agent_idx).get_action_and_value(observations, key, action_masks)

    def get_value_by_index(self, observations: env_types.Observation, agent_idx: int) -> chex.Array:
        return self._reconstruct_agent(agent_idx).get_value(observations)

    def maybe_resample(self, active_agent_idx: int, episode_done: bool, key: chex.PRNGKey) -> Tuple[int, chex.PRNGKey]:
        """Resample active agent index when an episode ends. Inactive slots (logit=-inf) are never sampled."""
        key, sample_key = jax.random.split(key)
        new_idx = jax.random.categorical(sample_key, self.mixture_logits.value)
        return jnp.where(episode_done, new_idx, active_agent_idx), key
