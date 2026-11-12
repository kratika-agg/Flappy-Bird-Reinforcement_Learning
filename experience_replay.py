
from collections import deque
import random


class ReplayMemory:
    """FIFO experience replay buffer with reproducible sampling."""

    def __init__(self, maxlen, seed=None):

        if isinstance(maxlen, bool) or not isinstance(maxlen, int):
            raise ValueError("Replay memory capacity must be an integer.")

        if maxlen <= 0:
            raise ValueError("Replay memory capacity must be greater than zero.")

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Seed must be a non-negative integer or None.")

        self.memory = deque(maxlen=maxlen)
        self.random_generator = random.Random(seed)

    def append(self, new_exp):

        if not isinstance(new_exp, (tuple, list)):
            raise TypeError("Experience must be a tuple or list.")

        if len(new_exp) != 5:
            raise ValueError(
                "Experience must contain exactly 5 elements: "
                "state, action, next_state, reward, terminated."
            )

        self.memory.append(tuple(new_exp))

    def sample(self, batch_size):

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("Batch size must be an integer.")

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if batch_size > len(self.memory):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"from memory containing {len(self.memory)} experiences."
            )

        return self.random_generator.sample(
            list(self.memory),
            batch_size
        )

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return (
            f"ReplayMemory(size={len(self.memory)}, "
            f"capacity={self.memory.maxlen})"
        )
