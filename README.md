# 🐦 Flappy Bird AI | Deep Reinforcement Learning

**Training an intelligent agent to play Flappy Bird using Deep Q-Networks (DQN).**

A hands-on implementation of Deep Reinforcement Learning, exploring how an AI agent learns optimal actions through environmental interaction, experience replay, and reward-based learning.

---

## 📌 Overview

This project focuses on developing an AI agent capable of learning to play the Flappy Bird game using **Deep Q-Learning (DQN)**.

The agent interacts with the game environment, observes the current state, selects actions, and learns from the rewards received. Through repeated interactions and neural network optimization, the agent gradually learns a playing strategy.

The implementation uses PyTorch for neural network operations and Gymnasium for environment interaction.

## 🧠 Core Concepts

The project explores the following Deep Reinforcement Learning concepts:

- **Deep Q-Network (DQN):** Approximating action-value functions using neural networks.
- **Experience Replay:** Storing and sampling past experiences for training.
- **Epsilon-Greedy Strategy:** Balancing exploration and exploitation.
- **Target Network:** Stabilizing Q-value learning.
- **Bellman Equation:** Updating estimated action values.
- **Mini-Batch Optimization:** Training the network using sampled experiences.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Neural network implementation |
| Gymnasium | Reinforcement learning environment |
| Flappy Bird Gymnasium | Flappy Bird simulation |
| PyYAML | Hyperparameter configuration |
| Pygame | Game rendering and interaction |

## 📁 Project Structure

```text
Flappy-Bird-Reinforcement-Learning/
│
├── agent.py
├── dqn.py
├── experience_replay.py
├── game_flappy_bird.py
├── parameters.yaml
├── README.md
└── .gitignore
```

## ⚙️ Implementation Details

### 1. Environment Setup
- Initializing the Flappy Bird environment.
- Observing game states and available actions.
- Interacting with the environment through Gymnasium.

### 2. DQN Architecture
- Initializing the policy network.
- Initializing the target network.
- Estimating Q-values for available actions.

### 3. Agent Implementation
- Loading hyperparameters from a YAML configuration file.
- Implementing epsilon-greedy action selection.
- Managing agent-environment interactions.
- Tracking episode rewards.

### 4. Experience Replay
- Storing state transitions.
- Sampling mini-batches from replay memory.
- Using past experiences to train the network.

### 5. Network Optimization
- Calculating target Q-values.
- Computing prediction loss.
- Updating network parameters using an optimizer.
- Synchronizing the target network.

## 📊 Project Progress

<escape>🚧</escape> **Currently in Development**

The project is being developed incrementally.

- [x] Flappy Bird environment setup
- [x] Initial DQN agent structure
- [x] Hyperparameter configuration
- [x] Experience replay implementation
- [x] Policy and target network initialization
- [x] Epsilon-greedy action selection
- [x] Initial optimization logic
- [ ] Complete and debug the training pipeline
- [ ] Train the agent
- [ ] Evaluate learning performance
- [ ] Visualize training results

## 🎯 Future Objectives

- Train the agent to play Flappy Bird autonomously.
- Experiment with different hyperparameter configurations.
- Analyze reward progression across episodes.
- Evaluate the trained agent's performance.
- Document training experiments and observations.

## 📚 Learning Journey

This project is part of my hands-on learning journey in Reinforcement Learning and Deep Learning.

The implementation follows a structured educational course, with practical exploration of DQN architecture, agent design, experience replay, and model optimization.

## 👩‍💻 Author

**Kratika Agrawal**

GitHub: [@kratika-agg](https://github.com/kratika-agg)

---

*Built with Python, PyTorch, and curiosity about intelligent systems.*

⭐ If you find this project interesting, consider giving the repository a star!