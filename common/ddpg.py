# --- ddpg.py ---
from common.value_networks import *
from common.policy_networks import *
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class DDPG():
    def __init__(self, args, replay_buffer, state_space, action_space):
        device_idx = args.device_idx
        if device_idx >= 0:
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print("Total device: ", self.device)

        self.replay_buffer = replay_buffer
        self.hidden_1 = args.hidden_1
        self.hidden_2 = args.hidden_2
        self.hidden_3 = args.hidden_3

        self.q_criterion = nn.MSELoss()
        q_lr = args.rate
        policy_lr = args.prate
        self.update_cnt = 0
        # 수정: args에서 받아온 τ 값을 사용 (기존에 0.0001로 고정되어 있었음)
        self.tau = args.tau  
        self.discount = args.discount
        self.GAMMA = self.discount

        self.actor = DPG_PolicyNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)
        self.critic_1 = QNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)
        self.critic_2 = QNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)
        self.target_critic_1 = QNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)
        self.target_critic_2 = QNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.target_actor = DPG_PolicyNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)

        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr=q_lr)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr=q_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=policy_lr)

        # --- 자동 엔트로피 조정 ---
        self.target_entropy = -np.prod(action_space.shape).item()
        self.log_alpha = torch.tensor(np.log(0.3), requires_grad=True, device=self.device)  # 초기값 ALPHA=0.3
        self.alpha_optim = optim.Adam([self.log_alpha], lr=policy_lr)

    def update_target_network(self, tau): # 타켓 네트워크 업데이트 비율을 10배 늘림 >> 0.0001 >> 0.001
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, std = self.actor(state)
        if deterministic:
            action = mu
        else:
            action, _ = self.actor.sample_normal(mu, std)
        return action.detach().cpu().numpy()[0]

    def critic_learn(self, states, actions, q_targets):
        q1 = self.critic_1(states, actions)
        loss1 = F.mse_loss(q1, q_targets.detach())
        self.critic_1_opt.zero_grad()
        loss1.backward()
        self.critic_1_opt.step()

        q2 = self.critic_2(states, actions)
        loss2 = F.mse_loss(q2, q_targets.detach())
        self.critic_2_opt.zero_grad()
        loss2.backward()
        self.critic_2_opt.step()
        return loss1, loss2

    def actor_learn(self, states):
        mu, std = self.actor(states)
        actions, log_pdfs = self.actor.sample_normal(mu, std) # 재매개변수화 구현
        log_pdfs = log_pdfs.squeeze(-1)
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        q_min = torch.min(q1, q2).squeeze(-1)
        alpha = self.log_alpha.exp()

        actor_loss = (alpha * log_pdfs - q_min).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- 자동 엔트로피 학습 ---
        alpha_loss = -(self.log_alpha * (log_pdfs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return actor_loss

    def update(self, batch_size):
        self.update_cnt += 1
        state, global_state, action, last_action, reward, next_state, next_global_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)

        next_mu, next_std = self.actor(next_state)
        next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std)
        next_log_pdf = next_log_pdf.squeeze(-1)
        target_q1 = self.target_critic_1(next_state, next_actions)
        target_q2 = self.target_critic_2(next_state, next_actions)
        target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_pdf

        y_i = reward + self.GAMMA * target_q * (1 - done)

        q_loss1, q_loss2 = self.critic_learn(state, action, y_i)
        actor_loss = self.actor_learn(state)
        self.update_target_network(self.tau)

        return q_loss1.detach().cpu().numpy(), q_loss2.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.critic_1.state_dict(), f'{path}/critic1.pkl')
        torch.save(self.critic_2.state_dict(), f'{path}/critic2.pkl')
        torch.save(self.target_critic_1.state_dict(), f'{path}/target_q1.pkl')
        torch.save(self.target_critic_2.state_dict(), f'{path}/target_q2.pkl')
        torch.save(self.actor.state_dict(), f'{path}/policy.pkl')

    def load_model(self, path):
        self.critic_1.load_state_dict(torch.load(f'{path}/critic1.pkl', map_location=self.device))
        self.critic_2.load_state_dict(torch.load(f'{path}/critic2.pkl', map_location=self.device))
        self.target_critic_1.load_state_dict(torch.load(f'{path}/target_q1.pkl', map_location=self.device))
        self.target_critic_2.load_state_dict(torch.load(f'{path}/target_q2.pkl', map_location=self.device))
        self.actor.load_state_dict(torch.load(f'{path}/policy.pkl', map_location=self.device))
