
import flappy_bird_gymnasium
import gymnasium as gym
import random
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    REQUIRED_PARAMETERS = {
        "alpha",
        "gamma",
        "epsilon_init",
        "epsilon_min",
        "epsilon_decay",
        "replay_memory_size",
        "mini_batch_size",
        "reward_threshold",
        "network_sync_rate",
    }

    def __init__(self, param_set):
        self.param_set = param_set

        # Load and validate configuration.
        try:
            with open("parameters.yaml", "r", encoding="utf-8") as f:
                all_param_set = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Configuration file 'parameters.yaml' was not found."
            )
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML syntax in parameters.yaml: {exc}"
            ) from exc

        if not isinstance(all_param_set, dict) or not all_param_set:
            raise ValueError(
                "parameters.yaml must contain a non-empty dictionary "
                "of configurations."
            )

        if param_set not in all_param_set:
            available = ", ".join(str(key) for key in all_param_set)
            raise ValueError(
                f"Configuration '{param_set}' was not found. "
                f"Available configurations: {available}"
            )

        params = all_param_set[param_set]

        if not isinstance(params, dict):
            raise ValueError(
                f"Configuration '{param_set}' must be a dictionary."
            )

        missing = self.REQUIRED_PARAMETERS - params.keys()

        if missing:
            raise ValueError(
                f"Missing required parameters in '{param_set}': "
                f"{', '.join(sorted(missing))}"
            )

        self._validate_parameters(params)

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(
            RUNS_DIR, f"{self.param_set}.log"
        )
        self.MODEL_FILE = os.path.join(
            RUNS_DIR, f"{self.param_set}.pt"
        )

    @staticmethod
    def _validate_parameters(params):
        """Validate DQN hyperparameters before training."""

        numeric_parameters = [
            "alpha",
            "gamma",
            "epsilon_init",
            "epsilon_min",
            "epsilon_decay",
            "reward_threshold",
        ]

        for name in numeric_parameters:
            value = params[name]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"Parameter '{name}' must be a number."
                )

            if not float("-inf") < value < float("inf"):
                raise ValueError(
                    f"Parameter '{name}' must be finite."
                )

        if params["alpha"] <= 0:
            raise ValueError(
                "Parameter 'alpha' must be greater than zero."
            )

        if not 0 <= params["gamma"] <= 1:
            raise ValueError(
                "Parameter 'gamma' must be between 0 and 1."
            )

        if not 0 <= params["epsilon_min"] <= 1:
            raise ValueError(
                "Parameter 'epsilon_min' must be between 0 and 1."
            )

        if not 0 <= params["epsilon_init"] <= 1:
            raise ValueError(
                "Parameter 'epsilon_init' must be between 0 and 1."
            )

        if params["epsilon_min"] > params["epsilon_init"]:
            raise ValueError(
                "'epsilon_min' cannot be greater than 'epsilon_init'."
            )

        if not 0 < params["epsilon_decay"] <= 1:
            raise ValueError(
                "Parameter 'epsilon_decay' must be greater than 0 "
                "and at most 1."
            )

        if params["reward_threshold"] <= 0:
            raise ValueError(
                "Parameter 'reward_threshold' must be greater than zero."
            )

        integer_parameters = [
            "replay_memory_size",
            "mini_batch_size",
            "network_sync_rate",
        ]

        for name in integer_parameters:
            value = params[name]

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Parameter '{name}' must be an integer."
                )

            if value <= 0:
                raise ValueError(
                    f"Parameter '{name}' must be greater than zero."
                )

        if params["mini_batch_size"] > params["replay_memory_size"]:
            raise ValueError(
                "'mini_batch_size' cannot exceed 'replay_memory_size'."
            )

    def run(self, is_training=True, render=False):

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            best_reward = float("-inf")

        else:
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE, map_location=device)
            )
            policy_dqn.eval()

        try:
            for episode in itertools.count():

                state, _ = env.reset()
                state = torch.tensor(
                    state, dtype=torch.float, device=device
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
                                policy_dqn(state.unsqueeze(dim=0))
                                .squeeze()
                                .argmax()
                            )

                    next_state, reward, terminated, truncated, _ = (
                        env.step(action.item())
                    )

                    episode_reward += reward

                    reward = torch.tensor(
                        reward, dtype=torch.float, device=device
                    )

                    next_state = torch.tensor(
                        next_state, dtype=torch.float, device=device
                    )

                    if is_training:
                        memory.append(
                            (state, action, next_state, reward, terminated)
                        )
                        steps += 1

                    state = next_state

                if is_training:
                    print(
                        f"episode={episode + 1} "
                        f"with total reward={episode_reward} "
                        f"& epsilon={epsilon}"
                    )
                else:
                    print(
                        f"episode={episode + 1} "
                        f"with total reward={episode_reward}"
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

                        with open(self.LOG_FILE, "a") as f:
                            f.write(log_msg + "\n")

                        torch.save(
                            policy_dqn.state_dict(),
                            self.MODEL_FILE
                        )

                        best_reward = episode_reward

                if is_training and len(memory) > self.mini_batch_size:

                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(
                        mini_batch,
                        policy_dqn,
                        target_dqn
                    )

                    if steps > self.network_sync_rate:
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
            terminations, dtype=torch.float, device=device
        )

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        current_q = (
            policy_dqn(states)
            .gather(dim=1, index=actions.unsqueeze(dim=1))
            .squeeze()
        )

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or test mode."
    )

    parser.add_argument(
        "hyperparameters",
        help="Hyperparameter configuration name"
    )

    parser.add_argument(
        "--train",
        help="Training mode",
        action="store_true"
    )

    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
