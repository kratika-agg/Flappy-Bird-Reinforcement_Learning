Flappy Bird AI | Deep Reinforcement Learning (DQN)

An AI agent that learns to play Flappy Bird using Deep Q-Networks (DQN) and PyTorch. The project demonstrates reinforcement learning through environment interaction, reward-based learning, and neural network optimization.

Tech Stack

* Language: Python
* Deep Learning: PyTorch
* Environment: Gymnasium, Flappy Bird Gymnasium
* Game Interface: Pygame
* Configuration: YAML

Key Concepts

* Deep Q-Network (DQN)
* Experience Replay
* Target Network
* Epsilon-Greedy Exploration
* Bellman Equation
* Model Checkpointing

Project Structure

Flappy-Bird-Reinforcement-Learning/
│
├── agent.py
├── dqn.py
├── experience_replay.py
├── game_flappy_bird.py
├── parameters.yaml
├── README.md
└── .gitignore

Local Setup and Installation

Follow these steps to run the project locally.

1. Clone the Repository

git clone https://github.com/kratika-agg/Flappy-Bird-Reinforcement-Learning.git
cd Flappy-Bird-Reinforcement-Learning

2. Create a Virtual Environment

python -m venv .venv

3. Activate the Virtual Environment

Windows (PowerShell):

.\.venv\Scripts\Activate.ps1

macOS / Linux:

source .venv/bin/activate

4. Install Dependencies

python -m pip install --upgrade pip
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml

Running the Project

1. Launch the Game

python game_flappy_bird.py

Use the SPACE key to control the bird.

2. Train the DQN Agent

python agent.py <hyperparameter_set> --train

Replace <hyperparameter_set> with a valid configuration key defined in parameters.yaml.

3. Evaluate the Trained Agent

python agent.py <hyperparameter_set>

Run evaluation after the trained model checkpoint has been generated.

Model Training

The agent was trained using the DQN training pipeline. Model checkpoints and training logs are generated locally in the runs/ directory and excluded from version control.

Author

Kratika Agrawal

GitHub Profile
