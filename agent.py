 
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
import atexit


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

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

    def run(self, is_training=True, render=False):

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        # Ensure the environment is closed when Python exits.
        atexit.register(env.close)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)

            # Copy weights from policy network to target network.
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            best_reward = float("-inf")

        else:
            # Load the best trained policy.
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

                    # Exploration or exploitation.
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

                    # Take an action in the environment.
                    next_state, reward, terminated, truncated, _ = (
                        env.step(action.item())
                    )

                    episode_reward += reward

                    # Convert reward and next state into tensors.
                    reward = torch.tensor(
                        reward, dtype=torch.float, device=device
                    )

                    next_state = torch.tensor(
                        next_state, dtype=torch.float, device=device
                    )

                    if is_training:
                        # Store true termination separately from truncation.
                        memory.append(
                            (state, action, next_state, reward, terminated)
                        )
                        steps += 1

                    state = next_state

                # Episode summary.
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

                    # Decay exploration rate.
                    epsilon = max(
                        epsilon * self.epsilon_decay,
                        self.epsilon_min
                    )

                    # Save the best-performing model.
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

                # Optimize and synchronize networks.
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
            # Always close the environment, including on exceptions.
            env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get batch of experiences.
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

        # Calculate target Q-values.
        # True terminations do not bootstrap; truncations still do.
        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        # Calculate predicted Q-values.
        current_q = (
            policy_dqn(states)
            .gather(dim=1, index=actions.unsqueeze(dim=1))
            .squeeze()
        )

        # Compute loss.
        loss = self.loss_fn(current_q, target_q)

        # Optimize the policy network.
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
