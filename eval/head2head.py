"""Head-to-head evaluation between two trained agents.

Loads the final checkpoint from each directory and plays them head-to-head.
Agent 1 plays as player 0, agent 2 as player 1.

Usage:
    uv run eval/head2head.py <dir1> <dir2> --env <env_name>

Example:
    uv run eval/head2head.py checkpoints/kuhn_poker/nash_pg/run0 \\
                             checkpoints/kuhn_poker/mmd/run0 \\
                             --env kuhn_poker
"""

import os
import logging
import warnings
import argparse
from functools import partial
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("orbax").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Sharding info.*")

import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import nnx

from envs import create_env
from agents import create_agent, BaseAgent


def find_final_step(checkpoint_dir: Path) -> int:
    ckpts = sorted(
        (d for d in checkpoint_dir.glob("checkpoint_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return int(ckpts[-1].name.split("_")[1])


def load_agent(checkpoint_dir: Path, env_name: str, key: chex.PRNGKey) -> BaseAgent:
    step = find_final_step(checkpoint_dir)
    key, subkey = jax.random.split(key)
    return create_agent(env_name, subkey).load_checkpoint(str(checkpoint_dir), step, key)


@partial(nnx.jit, static_argnames=("env", "num_envs", "max_steps"))
def evaluate(
    agent1: BaseAgent,
    agent2: BaseAgent,
    env,
    key: chex.PRNGKey,
    num_envs: int,
    max_steps: int,
) -> tuple[chex.Array, chex.Array]:
    """Run num_envs parallel episodes; return (ep_rewards, done) shape (num_envs,).

    ep_rewards[i]: cumulative reward for agent1 (player-0 slot) in episode i.
    done[i]:       True iff the episode completed within max_steps steps.
    """
    key, reset_key = jax.random.split(key)
    env_states, timesteps = jax.vmap(env.reset)(jax.random.split(reset_key, num_envs))

    def scan_step(carry, _):
        env_states, timesteps, ep_rewards, done, key = carry
        key, act_key = jax.random.split(key)

        act1, _, _ = agent1.get_action_and_value(
            timesteps.observation, act_key, timesteps.action_mask
        )
        act2, _, _ = agent2.get_action_and_value(
            timesteps.observation, act_key, timesteps.action_mask
        )
        actions = jnp.where(timesteps.current_player == 0, act1, act2)

        new_states, new_ts = jax.vmap(env.step)(env_states, actions)
        new_ep_rewards = jnp.where(~done, ep_rewards + new_ts.reward[:, 0], ep_rewards)
        new_done = done | new_ts.done

        return (new_states, new_ts, new_ep_rewards, new_done, key), None

    init = (
        env_states,
        timesteps,
        jnp.zeros(num_envs),
        jnp.zeros(num_envs, dtype=jnp.bool_),
        key,
    )
    (_, _, ep_rewards, done, _), _ = jax.lax.scan(scan_step, init, None, length=max_steps)
    return ep_rewards, done


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dir1", type=Path, help="Checkpoint dir for agent 1 (player 0)")
    parser.add_argument("dir2", type=Path, help="Checkpoint dir for agent 2 (player 1)")
    parser.add_argument("--env", required=True, help="Environment name (e.g. kuhn_poker)")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    key = jax.random.key(args.seed)

    key, k1, k2 = jax.random.split(key, 3)
    agent1 = load_agent(args.dir1, args.env, k1)
    agent2 = load_agent(args.dir2, args.env, k2)

    env = create_env(args.env)
    print(f"Env:     {args.env}")
    print(f"Agent 1: {args.dir1}")
    print(f"Agent 2: {args.dir2}")

    key, eval_key = jax.random.split(key)
    ep_rewards, done = evaluate(agent1, agent2, env, eval_key, args.num_envs, args.max_steps)

    ep_rewards_np = np.array(ep_rewards)
    done_np = np.array(done)
    n_done = int(done_np.sum())

    if n_done == 0:
        print("WARNING: no episodes completed within max_steps")
        return
    if n_done < args.num_envs:
        print(f"WARNING: only {n_done}/{args.num_envs} episodes completed")

    mean_r = float(ep_rewards_np[done_np].mean())
    std_r = float(ep_rewards_np[done_np].std())
    print(f"\nAgent 1 mean reward: {mean_r:+.4f} ± {std_r:.4f}  ({n_done} episodes)")


if __name__ == "__main__":
    main()
