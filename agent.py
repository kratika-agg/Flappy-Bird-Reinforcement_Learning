
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

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if env is not None:
            env.action_space.seed(self.seed)

    # Save a portable and safer model checkpoint.
    def save_checkpoint(self, model):

        checkpoint = {
            "format_version": 1,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        }

        temp_file = self.MODEL_FILE + ".tmp"

        try:
            torch.save(checkpoint, temp_file)

            # Replace the previous checkpoint only after saving succeeds.
            os.replace(temp_file, self.MODEL_FILE)

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Load a checkpoint safely and support older model files.
    def load_checkpoint(self, model):

        if not os.path.exists(self.MODEL_FILE):
            raise FileNotFoundError(
                f"Model file not found: {self.MODEL_FILE}. "
                "Train the agent first."
            )

        try:
            checkpoint = torch.load(
                self.MODEL_FILE,
                map_location=device,
                weights_only=True
            )

        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                self.MODEL_FILE,
                map_location=device
            )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid checkpoint: expected a dictionary."
            )

        # Support both the new checkpoint format and older state dictionaries.
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint

        if not isinstance(model_state, dict):
            raise ValueError(
                "Invalid checkpoint: model weights are missing."
            )

        if not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in model_state.items()
        ):
            raise ValueError(
                "Invalid checkpoint: model weights have an unsupported format."
            )

        model.load_state_dict(model_state)

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

                self.load_checkpoint(policy_dqn)
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

                        memory.append((
                            state,
                            action,
                            next_state_tensor,
                            reward_tensor,
                            terminated
                        ))

                        steps += 1

                    state = next_state_tensor

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

                    epsilon = max(
                        epsilon * self.epsilon_decay,
                        self.epsilon_min
                    )

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

                        # Save using the new checkpoint method.
                        self.save_checkpoint(policy_dqn)

                        best_reward = episode_reward

                    if len(memory) >= self.mini_batch_size:

                        mini_batch = memory.sample(
                            self.mini_batch_size
                        )

                        self.optimize(
                            mini_batch,
                            policy_dqn,
                            target_dqn
                        )

                    if steps >= self.network_sync_rate:

                        target_dqn.load_state_dict(
                            policy_dqn.state_dict()
                        )

                        steps = 0

        finally:

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

        with torch.no_grad():

            target_q = rewards + (
                (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        current_q = policy_dqn(states).gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze(dim=1)

        loss = self.loss_fn(
            current_q,
            target_q
        )

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
