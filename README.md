# NashPG: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria

This repository contains the implementation and experiments code for the paper "NashPG: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria", published in Transactions on Machine Learning Research (2026).

## Environments

JAX-based two-player zero-sum game environments:

- **Kuhn Poker** (`envs/kuhn_poker.py`) — Simplified poker with 3 cards [[Wikipedia](https://en.wikipedia.org/wiki/Kuhn_poker)]
- **Leduc Poker** (`envs/leduc_poker.py`) — Two-round poker with a public card [[OpenSpiel](https://github.com/google-deepmind/open_spiel)]
- **No-Limit Head-Up Texas Hold'em** (`envs/head_up_poker/`) — Full no-limit poker, two players [[Wikipedia](https://en.wikipedia.org/wiki/Texas_hold_%27em)]
- **Liar's Dice** (`envs/liar_dice.py`) — Single-hand variant with 5 dice and 6 sides [[Wikipedia](https://en.wikipedia.org/wiki/Liar%27s_dice)]
- **Dark Hex 3** (`envs/dark_hex3.py`) — 3×3 hex where players cannot see opponent's stones; classic and abrupt variants [[OpenSpiel](https://github.com/google-deepmind/open_spiel)]
- **Phantom Tic-Tac-Toe** (`envs/phantom_tictactoe.py`) — Tic-tac-toe with hidden opponent moves; classic and abrupt variants [[OpenSpiel](https://github.com/google-deepmind/open_spiel)]
- **Battleship** (`envs/battleship/`) — Classic naval strategy game [[Wikipedia](https://en.wikipedia.org/wiki/Battleship_(game))]

Each environment has a corresponding agent in `agents/`.

## Installation

We use [uv](https://github.com/astral-sh/uv) as our package manager:
```bash
uv sync
```

### exp-a-spiel (eas)

Exact exploitability for Dark Hex 3 and Phantom Tic-Tac-Toe requires [exp-a-spiel (eas)](https://github.com/gabrfarina/exp-a-spiel). This can be installed by cloning the repo and run `uv add ./exp-a-spiel`.

## Training

We use [Hydra](https://hydra.cc/docs/intro/) for configuration. Set `agent` and `env` to the game name (they match). See `conf/` for all available options.

### Nash Policy Gradient [[NashPG](https://arxiv.org/abs/2510.18183)]
```bash
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/nash_pg"
```

### Regularized Nash Dynamics [[R-NaD](https://arxiv.org/abs/2206.15378)]
```bash
uv run train/rnad.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/rnad"
```

### Magnetic Mirror Descent [[MMD](https://openreview.net/pdf?id=DpE5UYUQzZH)]
```bash
uv run train/mmd.py \
    algorithm.num_update=10000 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/mmd"
```

### Policy Space Response Oracles [[PSRO](https://arxiv.org/pdf/1711.00832)]
```bash
uv run train/psro.py \
    --config-name psro \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/psro"
```

### Neural Fictitious Self-Play [[NFSP](https://arxiv.org/pdf/1603.01121)]
```bash
uv run train/psro.py \
    --config-name nfsp \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/nfsp"
```

## Evaluation

### Exploitability (RL-based, for large games)

Trains a best-response agent via RL against a fixed checkpoint. Works for any environment.

```bash
uv run eval/compute_exploitability_rl.py \
    --checkpoint-dir checkpoints/liar_dice/nash_pg/run0 \
    --env liar_dice
```

Use `--step N` to evaluate a specific checkpoint (default: final). Use `--num-updates` to control training length.

### Exploitability (exact, game-tree based)

```bash
# Kuhn Poker (via OpenSpiel)
uv run eval/compute_exploitability_openspiel_kuhn.py \
    --checkpoint-dir checkpoints/kuhn_poker/nash_pg/run0 --step 1000

# Leduc Poker (via OpenSpiel)
uv run eval/compute_exploitability_openspiel_leduc.py \
    --checkpoint-dir checkpoints/leduc_poker/nash_pg/run0 --step 1000

# Dark Hex 3 (via eas)
uv run eval/compute_exploitability_eas_dark_hex3.py \
    --checkpoint-dir checkpoints/dark_hex3_abrupt/nash_pg/run0 --step 1000 --variant abrupt

# Phantom Tic-Tac-Toe (via eas)
uv run eval/compute_exploitability_eas_phantom_ttt.py \
    --checkpoint-dir checkpoints/phantom_tic_tac_toe_abrupt/nash_pg/run0 --step 1000 --variant abrupt
```

### Head-to-Head Evaluation

Plays two trained agents against each other and reports mean reward for agent 1:

```bash
uv run eval/head2head.py checkpoints/kuhn_poker/nash_pg/run0 \
                         checkpoints/kuhn_poker/mmd/run0 \
                         --env kuhn_poker
```

## Repository Structure

```
nash_policy_gradient/
├── envs/          # Game environment implementations
├── agents/        # Neural network agent architectures
├── train/         # Training algorithms (nash_pg, rnad, mmd, psro)
├── eval/          # Evaluation scripts (exploitability, head-to-head)
├── conf/          # Hydra configuration files
└── wrappers/      # Environment wrappers (auto-reset)
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{yu2026nashpgpolicygradientmethod,
      title={NashPG: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria}, 
      author={Eason Yu and Tzu Hao Liu and Clément L. Canonne and Yunke Wang and Chang Xu and Nguyen H. Tran and Stefano V. Albrecht},
      year={2026},
      eprint={2510.18183},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.18183}, 
}
```
