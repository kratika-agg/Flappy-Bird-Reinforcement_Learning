from collections import deque
import random

class ReplayMemory():
    #create FIFO=Experience Replay
    def __init__(self,maxlen,seed=None):
        self.memory=deque([],maxlen=maxlen)

    def append(self,new_exp):
        self.memory.append(new_exp)

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    #curr buffer size
    def __len__(self):
        return len(self.memory)

