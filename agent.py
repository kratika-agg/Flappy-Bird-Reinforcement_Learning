
import argparse
import itertools
import math
import os
import random

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from dqn import DQN
from experience_replay import ReplayMemory


# Select the best available device.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    REQUIRED_PARAMETERS = [
        "epsilon_init",
        "epsilon_min",
        "epsilon_decay",
        "replay_memory_size",
        "mini_batch_size",
        "network_sync_rate",
        "alpha",
        "gamma",
        "reward_threshold",
    ]

    def __init__(self, param_set):

        self.param_set = param_set

        # Load and validate YAML configuration.
        if not os.path.exists("parameters.yaml"):
            raise FileNotFoundError(
                "parameters.yaml was not found."
            )

        try:
            with open("parameters.yaml", "r", encoding="utf-8") as file:
                all_param_sets = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(
                "parameters.yaml contains invalid YAML."
            ) from error

        if not isinstance(all_param_sets, dict):
            raise ValueError(
                "The YAML file must contain a dictionary of configurations."
            )

        if param_set not in all_param_sets:
            raise ValueError(
                f"Configuration '{param_set}' was not found in parameters.yaml."
            )

        params = all_param_sets[param_set]

        if not isinstance(params, dict):
            raise ValueError(
                f"Configuration '{param_set}' must contain a dictionary."
            )

        missing_parameters = [
            key for key in self.REQUIRED_PARAMETERS
            if key not in params
        ]

        if missing_parameters:
            raise ValueError(
                "Missing configuration parameters: "
                + ", ".join(missing_parameters)
            )

        self.env_id = params.get("env_id", "FlappyBird-v0")

        # Validate numeric parameters.
        numeric_parameters = [
            "epsilon_init",
            "epsilon_min",
            "epsilon_decay",
            "alpha",
            "gamma",
            "reward_threshold",
        ]

        for key in numeric_parameters:
            value = params[key]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"'{key}' must be a finite number."
                )

        if not 0 <= params["epsilon_init"] <= 1:
            raise ValueError("epsilon_init must be between 0 and 1.")

        if not 0 <= params["epsilon_min"] <= 1:
            raise ValueError("epsilon_min must be between 0 and 1.")

        if params["epsilon_min"] > params["epsilon_init"]:
            raise ValueError(
                "epsilon_min cannot be greater than epsilon_init."
            )

        if not 0 < params["epsilon_decay"] <= 1:
            raise ValueError(
                "epsilon_decay must be greater than 0 and at most 1."
            )

        if not 0 <= params["gamma"] <= 1:
            raise ValueError("gamma must be between 0 and 1.")

        if params["alpha"] <= 0:
            raise ValueError("alpha must be greater than zero.")

        if params["reward_threshold"] <= 0:
            raise ValueError("reward_threshold must be greater than zero.")

        integer_parameters = [
            "replay_memory_size",
            "mini_batch_size",
            "network_sync_rate",
        ]

        for key in integer_parameters:
            value = params[key]

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"'{key}' must be an integer."
                )

            if value <= 0:
                raise ValueError(
                    f"'{key}' must be greater than zero."
                )

        if params["mini_batch_size"] > params["replay_memory_size"]:
            raise ValueError(
                "mini_batch_size cannot exceed replay_memory_size."
            )

        self.seed = params.get("seed", 42)

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer."
            )

        # Store validated parameters.
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.log"
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.pt"
        )

    def set_random_seeds(self, env=None):

        """Set random seeds for reproducible experiments."""

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if env is not None:
            env.action_space.seed(self.seed)

    def run(self, is_training=True, render=False):

        env = gym.make(
            self.env_id,
            render_mode="human" if render else None
        )

        try:

            self.set_random_seeds(env)

            num_states = env.observation_space.shape[0]
            num_actions = env.action_space.n

            policy_dqn = DQN(
                num_states,
                num_actions
            ).to(device)

            if is_training:

                # Create replay memory using the configured seed.
                memory = ReplayMemory(
                    self.replay_memory_size,
                    seed=self.seed
                )

                epsilon = self.epsilon_init

                target_dqn = DQN(
                    num_states,
                    num_actions
                ).to(device)

                target_dqn.load_state_dict(
                    policy_dqn.state_dict()
                )

                target_dqn.eval()

                steps = 0

                self.optimizer = optim.Adam(
                    policy_dqn.parameters(),
                    lr=self.alpha
                )

                best_reward = float("-inf")

            else:

                if not os.path.exists(self.MODEL_FILE):
                    raise FileNotFoundError(
                        f"Model file not found: {self.MODEL_FILE}. "
                        "Train the agent first."
                    )

                model_state = torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )

                policy_dqn.load_state_dict(model_state)
                policy_dqn.eval()

            for episode in itertools.count():

                state, _ = env.reset(
                    seed=self.seed + episode
                )

                state = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                )

                episode_reward = 0

                terminated = False
                truncated = False

                while (
                    not terminated
                    and not truncated
                    and episode_reward < self.reward_threshold
                ):

                    # Epsilon-greedy action selection.
                    if is_training and random.random() < epsilon:

                        action = env.action_space.sample()

                        action = torch.tensor(
                            action,
                            dtype=torch.long,
                            device=device
                        )

                    else:

                        with torch.no_grad():

                            action = (
                                policy_dqn(
                                    state.unsqueeze(dim=0)
                                )
                                .squeeze(dim=0)
                                .argmax()
                            )

                    # Take action in the environment.
                    next_state, reward, terminated, truncated, _ = (
                        env.step(action.item())
                    )

                    episode_reward += reward

                    reward_tensor = torch.tensor(
                        reward,
                        dtype=torch.float32,
                        device=device
                    )

                    next_state_tensor = torch.tensor(
                        next_state,
                        dtype=torch.float32,
                        device=device
                    )

                    if is_training:

                        # Store true termination separately from truncation.
                        memory.append((
                            state,
                            action,
                            next_state_tensor,
                            reward_tensor,
                            terminated
                        ))

                        steps += 1

                    state = next_state_tensor

                # Episode completed.
                if is_training:

                    print(
                        f"episode={episode + 1} "
                        f"with total reward={episode_reward:.2f} "
                        f"& epsilon={epsilon:.4f}"
                    )

                else:

                    print(
                        f"episode={episode + 1} "
                        f"with total reward={episode_reward:.2f}"
                    )

                if is_training:

                    # Decay exploration rate.
                    epsilon = max(
                        epsilon * self.epsilon_decay,
                        self.epsilon_min
                    )

                    # Save the best model.
                    if episode_reward > best_reward:

                        log_msg = (
                            f"best reward={episode_reward} "
                            f"for episode={episode + 1}"
                        )

                        with open(
                            self.LOG_FILE,
                            "a",
                            encoding="utf-8"
                        ) as file:
                            file.write(log_msg + "\n")

                        torch.save(
                            policy_dqn.state_dict(),
                            self.MODEL_FILE
                        )

                        best_reward = episode_reward

                    # Train the policy network.
                    if len(memory) >= self.mini_batch_size:

                        mini_batch = memory.sample(
                            self.mini_batch_size
                        )

                        self.optimize(
                            mini_batch,
                            policy_dqn,
                            target_dqn
                        )

                    # Synchronize target network.
                    if steps >= self.network_sync_rate:

                        target_dqn.load_state_dict(
                            policy_dqn.state_dict()
                        )

                        steps = 0

        finally:

            # Always close the environment.
            env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, next_states, rewards, terminations = zip(
            *mini_batch
        )

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)

        terminations = torch.tensor(
            terminations,
            dtype=torch.float32,
            device=device
        )

        # Calculate target Q-values.
        with torch.no_grad():

            target_q = rewards + (
                (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        # Calculate predicted Q-values.
        current_q = policy_dqn(states).gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze(dim=1)

        # Calculate loss.
        loss = self.loss_fn(
            current_q,
            target_q
        )

        # Update policy network.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or test the Flappy Bird DQN agent."
    )

    parser.add_argument(
        "hyperparameters",
        help="Name of the configuration in parameters.yaml"
    )

    parser.add_argument(
        "--train",
        help="Run the agent in training mode.",
        action="store_true"
    )

    args = parser.parse_args()

    dql = Agent(
        param_set=args.hyperparameters
    )

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
