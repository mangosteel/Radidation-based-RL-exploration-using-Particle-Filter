import gym
import time

env = gym.make('gym_ste_v4:MultiRotaryExtMatEnv-v0')
env.init_envs(seed=8201085478471, adj_act_on=True, pf_num=200, mean_on=True, gmm_num=3, kmeans_num=0, num_agents=2)

[global_state, comm_state] = env.reset()
env.uav[0].x = 15
env.uav[0].y = 30
env.uav[1].x = 45
env.uav[1].y = 30

i=1
while True:
    print("STEP: ", i)
    i += 1
    action=[0,1]
    [next_global_state, comm_next_state], reward, done, info = env.step(action)
    env.render(mode='human')
    time.sleep(0.1)
    if info[1] == True:
        print("COLLISION!!!!!!!!!!!!!!!!!")
        env.render(mode='human')
        time.sleep(10)
