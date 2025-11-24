import math
import random
import numpy as np
import torch

class ReplayBuffer:
    """ 
    Replay buffer for agent with GRU network additionally storing previous action, 
    initial input hidden state and output hidden state of GRU.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, global_state, action, last_action, reward, next_state, next_global_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, global_state, action, last_action, reward, next_state, next_global_state, done)
        # 아 이게 그러면 10만번 이상에서는 push가 일어난다는 것은 
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, gs_lst, a_lst, la_lst, r_lst, ns_lst, ngs_lst, d_lst=[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, global_state, action, last_action, reward, next_state, next_global_state, done = sample
            s_lst.append(state)
            gs_lst.append(global_state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            ngs_lst.append(next_global_state)
            d_lst.append(done)


        return s_lst, gs_lst, a_lst, la_lst, r_lst, ns_lst, ngs_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

