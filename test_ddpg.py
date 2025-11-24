'''
Recurrent Deterministic Policy Gradient (DDPG with GRU network)
Update with batch of episodes for each time, so requires each episode has the same length.
'''
import time
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

from common.buffers import *
from common.utils import *
from common.ddpg import *
from common.evaluator import *

import argparse
from gym import spaces

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("Learning device: ", device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RDPG')
    # Set Environment
    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_ste_v4:MultiRotaryExtMatEnv-v0', type=str, help='open-ai gym environment')

    # Set network parameter
    parser.add_argument('--hidden_1', default=256, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden_2', default=64, type=int, help='input num of GRU layer')
    parser.add_argument('--hidden_3', default=32, type=int, help='output num of GRU layer')
    parser.add_argument('--n_layers', default=1, type=int, help='number of stack for hidden layer')
    parser.add_argument('--rate', default=0.0002, type=float, help='Q learning rate')
    parser.add_argument('--prate', default=0.00002, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.95, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.0001, type=float, help='moving average for target network')
    parser.add_argument('--drop_prob', default=0.2, type=float, help='dropout_probability')

    # Set learning parameter
    parser.add_argument('--rbsize', default=100000, type=int, help='Memory size')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--blength', default=1, type=int, help='minibatch sequence length')
    parser.add_argument('--warmup', default=100000, type=int, help='warmup size (steps or episodes)')
    parser.add_argument('--max_episodes', default=200000, type=int, help='Number of episodes')
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode')
    parser.add_argument('--validate_episodes', default=20, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_interval', default=1000, type=int, help='Validation episode interval')
    parser.add_argument('--epsilon_rate', default=0.1, type=int, help='linear decay of exploration policy')

    #etc
    parser.add_argument('--pause_time', default=0, type=float, help='Pause time for evaluation')
    parser.add_argument('--model_path', default='model/low_noise_current/gym_ste_v4:MultiRotaryExtEnv-v0-run10/', type=str, help='Output root')
    parser.add_argument('--model_path_current', default='.', type=str, help='Output root')

    #CUDA GPU
    parser.add_argument('--device_idx', default=device_idx, help='cuda device num: -1(CPU), 0<= (GPU) ')


    args = parser.parse_args()

    env = gym.make(args.env)
    env.init_envs(seed=8201085478471, adj_act_on=False, pf_num=2000,crash_check_on=False, mean_on=True, gmm_num=3, kmeans_num=0, num_agents=1)

    action_space = env.action_space
    state_space  = env.observation_space

    replay_buffer_size = args.rbsize
    replay_buffer = ReplayBuffer(1)


    torch.autograd.set_detect_anomaly(True)
    alg = DDPG(args, replay_buffer, state_space, action_space)


    if args.mode == 'test':
        export_parameter_to_text(args, env)
        test_episodes = 1000
        max_steps=300

        alg.load_model(args.model_path)
        total_reward = 0
        total_steps = 0
        total_q_diff = 0
        success_episodes = 0

        steps_list = []
        q_diff_list = []
        episode_reward_list = []
        sim_time_mean = []
        mean_search_time = []

        for episode in range(test_episodes):
            env.close()
            [global_state, comm_state] = env.reset()
            #state = comm_state[0]
            #env.render_background(mode='human') #Draw background plot
            episode_steps = 0
            episode_reward = 0.
            sim_time = []

            done = False

            mst_start = time.time() # mean search time 기준
            steps = 0
            while not done:
                start = time.time()
                steps += 1

                # action = [alg.policy_net.get_action(comm_state[n], noise_scale=0.0) for n in range(env.num_agents)] # no noise for testing
                action = [alg.get_action(comm_state[n],deterministic=True) for n in range(env.num_agents)] # 액션은 numpy!  deterministic=True
                [next_global_state, comm_next_state], reward, done, info = env.step(action)
                comm_state = comm_next_state

                if steps >= max_steps: done = True
                # env.render(mode='human') #Draw simulation
                sim_time_temp = time.time()-start
                sim_time.append(sim_time_temp)
                # update
                episode_reward = reward[0]

                episode_steps += 1
                time.sleep(args.pause_time)
                if info[1] == True:
                    print("COLLISION!!!!!!!!!!!!!!!!!")
                    # env.render(mode='human')
                    time.sleep(10)

            mst = time.time() - mst_start
            mean_search_time.append(mst)
            sim_time_temp = np.mean(sim_time)
            sim_time_mean.append(sim_time_temp)

            q_diff = abs(env.radiation.S_mass-env.uav[0].mean_mass) / env.radiation.S_mass * 100 # 각 에피소드의 상대오차..
            prRed('[Evaluate] Episode_{:07d}: | mean_reward:{} | sim_time:{} | q_diff:{}'.format(episode, round(episode_reward,1), round(sim_time_temp,4), q_diff) )
            print("Episode steps: ", steps)

            total_reward += episode_reward
            if episode_reward >= 100:
                total_steps += steps
                total_q_diff += q_diff
                success_episodes += 1

            steps_list.append(steps)
            q_diff_list.append(q_diff)
            episode_reward_list.append(episode_reward)

        mean_search_time_val = np.mean(mean_search_time)
        mean_reward = total_reward/test_episodes
        prGreen('[Evaluate] Episode_{:07d}: mean_reward:{} mean_search time : {}'.format(episode, mean_reward,mean_search_time_val))
        mean_steps = total_steps/success_episodes
        mean_q_diff = total_q_diff/success_episodes
        print("Mean steps: ", mean_steps)
        print("Q_diff: ", mean_q_diff)

        results_list_dic = {"steps_list": steps_list, "episode_reward_list": episode_reward_list, "sim_time": sim_time_mean, "q_diff": q_diff_list}

        savemat('{}/test_data'.format(args.model_path)+".mat", results_list_dic)

