'''
Deep Deterministic Policy Gradient (DDPG)
Update with batch of episodes for each time, so requires each episode has the same length.
'''
import time
import math
import random

import gym
import numpy as np

import torch

from common.buffers import *
from common.utils import *
from common.ddpg import *
from common.evaluator import *

import argparse
from gym import spaces

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

GPU = True
device_idx = 0




if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("Learning device: ", device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDPG')
    # Set Environment
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_ste_v4:MultiRotaryExtEnv-v0', type=str, help='open-ai gym environment')

    # Set network parameter / 알고보니 속성값을 저장하는데 유용한 클래스 친구였네~
    parser.add_argument('--hidden_1', default=256, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden_2', default=64, type=int, help='input num of linear layer')
    parser.add_argument('--hidden_3', default=32, type=int, help='output num of linear layer')
    parser.add_argument('--n_layers', default=1, type=int, help='number of stack for hidden layer')
    parser.add_argument('--rate', default=0.0001, type=float, help='Q learning rate')
    parser.add_argument('--prate', default=0.00001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.95, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network') # 0.001 , 0.005, 0.01 / 0.001이 경우에 어제는 빨랐는데 오늘은 엄청 느려지고 로드도 커짐 왜 그럴까?? 애초에 신경망에서 무한대의 손실이 발생함..
    parser.add_argument('--drop_prob', default=0.2, type=float, help='dropout_probability')
    parser.add_argument('--action_bound', default=1, type=int, help='action_bound : [-1,1]')

    # Set learning parameter
    parser.add_argument('--rbsize', default=100000, type=int, help='Memory size')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--blength', default=1, type=int, help='minibatch sequence length')
    parser.add_argument('--warmup', default=100000, type=int, help='warmup size (steps)')
    #parser.add_argument('--max_episodes', default=100000, type=int, help='Number of episodes')
    parser.add_argument('--max_steps', default=15e6, type=int, help='Number of episodes')
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode')
    parser.add_argument('--validate_episodes', default=20, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_interval', default=1000, type=int, help='Validation episode interval')
    parser.add_argument('--epsilon_rate', default=0.1, type=int, help='linear decay of exploration policy')

    #etc
    parser.add_argument('--pause_time', default=0, type=float, help='Pause time for evaluation')
    parser.add_argument('--model_path', default='model/low_noise', type=str, help='Output root')
    parser.add_argument('--model_path_current', default='model/low_noise_current', type=str, help='Output root')

    #CUDA setting
    parser.add_argument('--device_idx', default=device_idx, help='cuda device num: -1(CPU), 0<= (GPU) ')


    args = parser.parse_args()

    env = gym.make(args.env) # args.env : gym_ste_v4:MultiRotaryExtEnv-v0
    
    # print() 그럼 우선 env를 구성하고 있는 가스확산모델과 파티클 필터, 그리고 여러 알고리즘(k_means, 혼합가우시안에 대한 대략적인 이해만/ 활용의 범주!)
    # 일단 나의 1차 목표는 그럼 일단 코드 모두를 이해하는 것 보다는 가스확산모델을 일단 이해하고 그것을 방사능 모델로 어떻게 바꿀지/ 입력변수나 출력변수에 대한 고민... 
    env.init_envs(seed=8201085478471, adj_act_on=False, crash_check_on=False, pf_num=200,lidar_on = False, mean_on=True, gmm_num=3, kmeans_num=0, num_agents=1)
    # 아 여기서 pf 개수와 gmm클러스터링 개수가 초기화되는 거구나!    / 파티클 개수를 200개로 하자.. 2000개라서 느릴지도?
    action_space = env.action_space
    state_space  = env.observation_space # base.py에서 정의된 obs space, action_space임!
    
    batch_size = args.bsize  # each sample in batch is an episode for linear policy (normally it's timestep)
    update_itr = 1  # update iteration
    n_layers = args.n_layers
    replay_buffer_size = args.rbsize
    replay_buffer = ReplayBuffer(replay_buffer_size)
    args.model_path = get_output_folder(args.model_path, args.env)
    args.model_path_current = get_output_folder(args.model_path_current, args.env)
    torch.autograd.set_detect_anomaly(False) # 이건 신경망의 이상치 디버깅 인데 그냥 끄고 해보자...

    alg = DDPG(args, replay_buffer, state_space, action_space)
    evaluate = Evaluator(args)
    box =[]
    k=0
    if args.mode == 'train': # 아마 에피소드 하나를 처음시작 할떄 초기화 하는 과정이고 매 에피소드마다 이럴 것....
        export_parameter_to_text(args, env)
        frame_idx   = 0
        total_steps = 0
        epsilon_steps = int(args.max_steps*0.75)
        rewards=[]
        q_loss_list1 =[]
        q_loss_list2 =[]
        highest_reward = -100 
        update_bool = False
        validate_start = 0
        i_episode = 0
        epi10_per_reward = []
        while total_steps <= args.max_steps: # 에피소드 하나의 과정!
            i_episode +=1
            episode_time = 0
            episode_steps = 0
            q_loss_list=[]  # 그럼 차라리 텐서보드로 받지 말고 이걸 그냥 넘파이 써서 csv 로 받으면 안되나? 절대경로 이용해서?...
            policy_loss_list=[]
            
            [global_state, comm_state] = env.reset() # 아마 단독 에이전트이면 comm_state : [] 일것..
            state = comm_state[0] # 여기서 상태 데이터 초기화 시작!
            last_action = [0]
            episode_reward = [] # 이것도 넘파이로 따로 저장한다면?.... headless로 해놓고 개 많이 돌리면 5000개정도는 금방될 듯?..
                               # 근데 이건 매 스텝마다 초기화 되므로 
            batch_length = 0

            if update_bool == False and len(replay_buffer) == args.rbsize: # 아마 버퍼에 있는 데이터가 배치 사이즈만큼 딱 찰때부터 업데이트가 가능하다 라는 것을 시사함!
                update_bool = True
                validate_start = i_episode
            
            if evaluate is not None and (i_episode-validate_start)%args.validate_interval == 0 and update_bool:
                policy = lambda x: alg.get_action(x)
                debug = True
                visualize = True
                validate_reward = evaluate(env, policy, i_episode, debug, visualize)
                if debug:
                    prRed('[Evaluate] Episode_{:04d}: mean_reward:{}'.format(i_episode, validate_reward)) # :07d
                # Save intermediate model
                if highest_reward < validate_reward:
                    prRed('Highest reward: {}, Validate_reward: {}'.format(highest_reward, validate_reward))
                    highest_reward = validate_reward
                    alg.save_model(args.model_path)
            
            for step in range(args.max_episode_length):
                state_sum = np.sum(state)
                if np.isnan(state_sum):
                    print("state: ", state)
                    time.sleep(5)
                start_time = time.time()
                noise_level = max((epsilon_steps - total_steps)/epsilon_steps, 0)
                
                

                action = [alg.get_action(comm_state[n]) for n in range(env.num_agents)] # 액션은 numpy!
                action = np.clip(action, -args.action_bound, args.action_bound) # 액션 클리핑! -1~1
                single_action = action[0] # 액션은 액터 신경망으로 얻기!

                #print("STEP: ", step)
                [next_global_state, comm_next_state], reward, done, info = env.step(action) # 여기서 상태 업데이트 시작!
                # print(f"reward : {reward}   done : {done}  ") #근데 왜 done이 trye인데 99.99보상을 계속받는걸까?
                # print()
                next_state = comm_next_state[0] # 근데 global_state가 mvg데이터 들어간거 아닌가? 데어터가 정상적으로 뜨네 내가 착각한건가?
                # print(f"comm_state : {next_state}")  # 에이전트가 1대일때는 global_state 와 comm_state가 서로 같다!
                # print()
                # print(f"global_state : {next_global_state}") /  어차피 이 과정자체는 버퍼에 데이터 채우기는 과정!!
                single_reward = reward[0]

                if batch_length==0: # 배치 사이즈가 없다면(초기값).. 각 상태 데이터에 해당하는 리스트들을 만듬!
                    batch_state = []
                    batch_global_state = []
                    batch_action = []
                    batch_last_action = []
                    batch_reward = []
                    batch_next_state = []
                    batch_next_global_state = []
                    batch_done = []

                batch_state.append(state)
                batch_global_state.append(global_state)
                batch_action.append(single_action)
                batch_last_action.append(last_action)
                batch_reward.append(single_reward)
                batch_next_state.append(next_state)
                batch_next_global_state.append(next_global_state)
                batch_done.append(done)

                # print(state == global_state)  # 어차피 단독 에이전트에서는 둘다 같아서 global을 안쓴 문제는 아닌거 같은데....
                # print()
                episode_reward.append(reward) # 리워드 저장하기(하나의 에피소드에서)
                
                # print(f'rew ;: {reward}')
                # print()
                state = next_state # 상태 업데이트!
                comm_state = comm_next_state
                global_state = next_global_state
                last_action = single_action
                frame_idx += 1
                total_steps += 1
                batch_length += 1
                episode_steps += 1
                
                if i_episode% 10 == 1:
                     
                    # env.render(mode='human') # 여기에다 배경까지 추가해봐?
                    # env.render_background_sample(mode='human') # 방사능 모델 시각화..
                    
                    # env.render_background(mode='human') # 가스모델 시각화..
                    pass

                if batch_length == args.blength: # 이건그냥 1번 쓰고 바로 버퍼에 넣는거 
                    batch_length = 0  # 그러면 이건 에피소드 한번 끝날때 마다 1개씩 넣는거임??? / 에피소드 마다 얻을 수 있는 샘플개수가 달라서...
                    replay_buffer.push(batch_state, batch_global_state, batch_action, batch_last_action, \
                                       batch_reward, batch_next_state, batch_next_global_state, batch_done)
#                    replay_buffer.get_length()

                if update_bool: # update_bool  
                    for _ in range(update_itr):
                        q_loss1,q_loss2, policy_loss = alg.update(batch_size) # 버퍼에 넣어넣고 여기서 업데이트를 진행하는듯?
                        q_loss_list1.append(q_loss1)  # 그럼 손실얻는 거는 또 update모듈에서 구현해서 원래의 구조를 그대로 가져가보자..
                        q_loss_list2.append(q_loss2) 
                        policy_loss_list.append(policy_loss)
                         
                    if i_episode % 20 == 0:
                        alg.save_model(args.model_path_current)
                        
                
                        
                
                if done:  # should not break for cases to make every episode with same length
                    batch_length = 0
                    break        

                episode_time = episode_time + time.time()-start_time
                # print(f'{replay_buffer.get_length()}') # 이거 10만개 안채워도 렌더링 되는데 차근차근 흠..?
                # print()
                #print("time per step: ",  time.time()-start_time)

            if info[1] == True:
                print("COLLISION!!!!!!!!!!!!!!!!!")
                # env.render(mode='human')
                # time.sleep(10)

            print("Time per step: ",  episode_time/episode_steps) # 이 부분을 조금더 간결하게?
            print('Eps: ', i_episode, '| Reward: ', np.sum(episode_reward,axis=0).tolist(),
                  '| Loss: ', np.average(q_loss_list1),np.average(q_loss_list2), np.average(policy_loss_list), 
                  ' | episode_steps: ', episode_steps, ' | total_steps: ', total_steps, ' | buffer length: ', replay_buffer.get_length())
            print("DONE [converge_done, crash_done, block_done, timeout_done] :", info, 'nearby:', env.nearby) # "" >> info자리
            
            
        
        

