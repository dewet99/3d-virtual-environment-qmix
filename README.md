# Cooperative Multi-Agent Reinforcement Learning in Sparse-Reward, Partially Observable 3D Environments with Curriculum-Transfer Learning
The work represented here was done for my MEng thesis. It consists of an environment based very loosely on the Portal 2 game. The environment was built in Unity, and the Unity MLAgents package was used to communicate between the environment and the Python trainer. None of the MLAgents trainers or alogirhtms were used, only the grpc protocols for communication.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Environments](#environments)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project implements the QMIX Multi Agent Reinforcement Learning algorithm in a 3D virtual environment. The algorithm is augmented with a number of features to improve its performance in the 3D environment.

## Features
The main features we implemented are as follows. These features aren't particularly groundbreaking, but they have not really been combined to see how well they work together in a multi-agent RL context.
- QMIX
- Distributed RL using Ray, splitting the RL loop into multiple executors and a learner process
- Recurrent burn-in
- Stored recurrent hidden state
- n-step returns
- Prioritised experience replay
- Reward standardisation
- Noisy neural networks for exploration

## Environments
We include the following environments and perform various experiments in them. Note that there is no environment for experiment_2, as experiment_1 and experiment_2 use the same environment with different configs.
- experiment_1_ablation
- experiment_3_sparse_rewards
- experiment_4_full_action_space_pre_train
- experiment_4_full_action_space
- experiment_5_cooperative

For each of the above environments, there is also a corresponding test environment. These environments end with "_test". Functionally, they are exactly the same, except that the test environments allow you to see the environment, both from a top-down perspective and from each agent's perspective. You can open a test environment by navigating to its respective .x86_64 file. Linux only.

### Environment Controls
The test environments can be controlled with heuristics. It's a bit janky, but it's not really meant for human players to solve. The controls are as follows:

#### Camera Controls:
Use the C button on your keyboard to change camera perspective. It has 6 different options, starting at 0 (on launching the environment) to 5.
- 0: On opening the environment. No actions are possible.
- 1: You are able to control the green agent, but you cannot see it's perspective. You have to press "C" again to see its perspective. See what I did there?
- 2: Starting agent perspective becomes active.
- 3: Top-down view, control green agent.
- 4: Top-down view, control blue agent.
- 5: Top-down view, no control.

#### Agent Controls
You control an agent as follows:
- W, A, S, D: forward, left, back, right, respectively.
- left arrow, right arrow: look left and right, respectively
- left-click: place portal A, in the environments which has this functionality enabled (experiments 4 and 5)
- right-click: place portal B, in the environments which has this functionality enabled (experiments 4 and 5)

## Installation

Provide instructions on how to install your project. Include any dependencies, prerequisites, or setup steps.

```bash
# Example installation command
git clone https://github.com/your-username/your-repository.git
cd your-repository
npm install  # or any other necessary installation command
