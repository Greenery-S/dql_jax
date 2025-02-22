# README

This repository is an implementation of the [Diffusion Policy for Offline RL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) algorithm using JAX/Flax. It also marks my first attempt at constructing a relatively complex reinforcement learning system using technologies beyond PyTorch.

## Installation

### 1. Install Standard Dependencies

Please install the following packages:

- **[D4RL](https://github.com/rail-berkeley/d4rl)**
- **[OpenAI Gym](https://github.com/openai/gym)**
- **[NumPy](https://numpy.org/)**
- **[JAX](https://github.com/google/jax)**
- **[Flax](https://github.com/google/flax)**
- **[Optax](https://github.com/deepmind/optax)**
- **[Chex](https://github.com/deepmind/chex)**
- **[Distrax](https://github.com/deepmind/distrax)**
- **[Weights & Biases](https://wandb.ai/)**

You can install them using the following command (adjust based on your environment):

```bash
pip install d4rl gym numpy jax flax optax chex distrax wandb
```

### 2. Clone & Locally Install jaxrl_m

Execute the following commands:

```bash
git clone https://github.com/dibyaghosh/jaxrl_m.git
cd jaxrl_m
pip install -e .
```

Using the `-e` parameter installs the package in development mode, ensuring that the project's dependency library code points to your local version of **jaxrl_m**.

## Running the Project

From the project's root directory, run:

```bash
python run_<algo_name>.py
```

## File Structure

- **`hyper_<name>.py`**: Contains default hyperparameters and tuning configurations.
- **`util_<name>.py`**: Includes utility functions for data loading, models, and other helper operations.
- **`model_<name>.py`**: Defines the network architecture.
- **`algo_<name>.py`**: Contains the core logic of the RL agent, including creation, updates, and sampling.
- **`run_<name>.py`**: The entry point for running the program.
- **`xxx_test.py`**: Test files.

## Training Results

The training process and results can be monitored on the **[Weights & Biases](https://wandb.ai/)** platform.

## Performance Report

Test data on an RTX 4060 gaming laptop:

- **Training Speed**: Increased from ~38 iterations per second to ~650 iterations per second, marking a significant speedup.
- **GPU Utilization**: Risen from ~20% to ~45%, a modest increase.
- **GPU Memory Usage**: Grew from ~15% to ~70%, ensuring more efficient GPU resource usage.

## Acknowledgements

- Thanks to **[JAX](https://github.com/google/jax)**, **[Flax](https://github.com/google/flax)**, **[Optax](https://github.com/deepmind/optax)**, **[Distrax](https://github.com/deepmind/distrax)**, and other high-quality deep learning libraries for their elegant code and comprehensive documentation.
- Appreciation goes to [jaxrl](https://github.com/ikostrikov/jaxrl), [jaxrl2](https://github.com/ikostrikov/jaxrl2), and **[jaxrl_m](https://github.com/dibyaghosh/jaxrl_m/tree/main)** for their outstanding contributions to applying JAX/Flax in reinforcement learning.
- Special thanks to the original author of *Diffusion Policy for Offline RL* for providing a robust algorithm that maintained high reproducibility even after migrating frameworks.

---

If you encounter any issues while using this project, please feel free to submit an issue or a pull request to help improve it. Wishing you success in your reinforcement learning research!