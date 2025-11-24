# --- value_networks.py ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from .initialize import *

class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:
            pass  
        self.activation = activation

    def forward(self):
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation):
        super().__init__(state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]

class QNetwork(QNetworkBase):
    """
    Q network with LSTM structure.
    """
    def __init__(self, device_idx, state_space, action_space, hidden_1, hidden_2, hidden_3, n_layers, drop_prob, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)

        if device_idx >= 0:
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print("Value device: ", self.device)

        self.linear1 = nn.Linear(self._state_dim + self._action_dim, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.linear3 = nn.Linear(hidden_2, hidden_3)
        self.linear4 = nn.Linear(hidden_3, 1)
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action):
        # 수정: device 이동은 to(self.device)로 수행 (이미 state와 action가 device에 있다면 중복 이동되지 않음)
        x = torch.cat([state, action], -1).to(self.device)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x
