# Flappy Bird AI | Deep Reinforcement Learning (DQN)

An AI agent that learns to play Flappy Bird using **Deep Q-Networks (DQN)** and **PyTorch**. This project demonstrates reinforcement learning through environment interaction, reward-based learning, and neural network optimization.

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Environment:** Gymnasium, Flappy Bird Gymnasium
- **Game Interface:** Pygame
- **Configuration:** YAML

## Key Concepts

- Deep Q-Network (DQN)
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
- Bellman Equation
- Model Checkpointing

## Project Structure

```text
Flappy-Bird-Reinforcement-Learning/
├── agent.py
├── dqn.py
├── experience_replay.py
├── game_flappy_bird.py
├── parameters.yaml
├── README.md
└── .gitignore
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/kratika-agg/Flappy-Bird-Reinforcement-Learning.git
cd Flappy-Bird-Reinforcement-Learning
```

### 2. Install Dependencies

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

### 3. Run the Game

```bash
python game_flappy_bird.py
```

### 4. Train the Agent

```bash
python agent.py <hyperparameter_set> --train
```

### 5. Evaluate the Trained Agent

```bash
python agent.py <hyperparameter_set>
```

Replace `<hyperparameter_set>` with a valid configuration key from `parameters.yaml`.

## Model Training

The agent was trained using the DQN pipeline. Model checkpoints and training logs are generated in the local `runs/` directory and excluded from version control.

## Author

**Kratika Agrawal**

[GitHub](https://github.com/kratika-agg)