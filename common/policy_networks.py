# --- policy_networks.py ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:
            pass  
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]

    def forward(self):
        pass

    def evaluate(self):
        pass 

    def get_action(self):
        pass

    def sample_action(self,):
        a = torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return a.numpy()

class DPG_PolicyNetwork(PolicyNetworkBase):
    """
    Deterministic policy gradient network with LSTM structure.
    """
    def __init__(self, device_idx, state_space, action_space, hidden_1, hidden_2, hidden_3, n_layers, drop_prob, init_w=3e-3):
        super().__init__(state_space, action_space)

        if device_idx >= 0:
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print("Policy device: ", self.device)

        self.linear1 = nn.Linear(self._state_dim, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.linear3 = nn.Linear(hidden_2, hidden_3)
        self.mu = nn.Linear(hidden_3, self._action_dim)
        self.std = nn.Linear(hidden_3, self._action_dim)
        
        with torch.no_grad():
            self.mu.weight.uniform_(-init_w, init_w)
            self.mu.bias.uniform_(-init_w, init_w)
            self.std.weight.uniform_(-init_w, init_w)
            self.std.bias.uniform_(-init_w, init_w)

        self.std_bound = [1e-2, 1.0]
        self.action_bound = 1

    def forward(self, state):
        activation = F.relu
        x = activation(self.linear1(state))
        x = activation(self.linear2(x))
        x = activation(self.linear3(x))
        mu = torch.tanh(self.mu(x)) 
        std = F.softplus(self.std(x)) + 1e-3
        return mu, std
       
    def sample_normal(self, mu, std):
        # 수정: reparameterization 적용 / 아마 이 부 부분에서 문제가 발생했었던것 같다.
        normal = Normal(mu, std)
        x_t = normal.rsample()  # reparameterized sample (gradient 지원) / 원래는 normal.sample이였음 이건 미분이 안되나??
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # tanh 변환에 대한 보정
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)  # 우리가 x라는 원래 값을 가지고 결과를 만들었는데, 
        #그 결과가 tanh라는 함수를 거치면서 "꼬이게" 돼.그래서 "원래의 확률"에서 "tanh 때문에 꼬인 정도"를 빼줘야 결과가 올바르게 계산돼.
        #이 과정을 "로그 확률 보정"이라고 부르는 거야.
        return action, log_prob

    def evaluate(self, state, noise_scale=0.0):
        normal = Normal(0, 1)
        mu, std = self.forward(state)
        noise = noise_scale * normal.sample(mu.shape).to(self.device)
        x_t = mu + noise * std
        action = torch.tanh(x_t)
        return action

    def get_action2(self, state, noise_scale=1.0):
        # 수정: 불필요한 while 루프를 제거하고 벡터화된 연산 사용
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, std = self.forward(state)
        normal = Normal(torch.zeros_like(mu), torch.ones_like(std))
        x_t = mu + noise_scale * std * normal.sample()
        action = torch.tanh(x_t)
        return action.detach().cpu().numpy()[0]
