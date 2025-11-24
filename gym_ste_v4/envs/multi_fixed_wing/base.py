import sys
import copy
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
import pyglet
from matplotlib import cm

from gym_ste_v4.envs.pf_mvg_conv_multi.agent import *

from gym.envs.registration import register
from datetime import datetime
import time

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self): # necessary to draw somthing
        self.label.draw()

class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        self.normalization = True
    
        #---------------------- important settings ---------------------
        self.env_sig = 0.2 #0.1 #0.2 #1.0 #0.2 #0.4
        self.sensor_sig_m = 0.1 #0.05 #0.1 #0.5 #0.1 #0.2
        self.agent_v = 4                # 2m/s
        self.delta_t = 1                # 1sec
        self.wifi_range = 10    # 10m
        self.crash_warning_range = 8
        self.crash_range = 4

        self.success_reward        = +100
        self.crash_warning_penalty = +0 #-0.1
        self.crash_penalty         = -100 #-100 or 0 for disable
        self.comm_reward           = +0 #+0.01
        self.step_penalty          = -0.01
        self.renew_reward          = +0.01
        #--------------------------Common Env----------------------------
        self.debug = False

        self.last_measure = 0
        self.court_lx = 60              # the size of the environment
        self.court_ly = 60              # the size of the environment
        self.max_step = 150

        # gas sensing
        self.conc_eps = 0.2 # minimum conc
        self.conc_max = 100

        # rendering
        self.screen_height = 200
        self.screen_width = 200
        self.viewer = None                  # viewer for render()
        self.background_viewer = None       # viewer for background
        self.scale = self.screen_width/self.court_lx
        self.true_conc = np.zeros((self.court_lx, self.court_ly))

        #--------------------------Initial data-------------------------
        self.gas_measure = -1;
        self.outborder = False          # in search area

        self.total_time = 0.
        self.update_count = 0

        self.CovXxp = 0.
        self.CovXyp = 0.
        self.CovXqp = 0.
        self.warning = False

        self.max_q = 5000
        #------------------------ Action space ------------------------
        self.action_angle_low  = -1
        self.action_angle_high =  1
        self.action_space = spaces.Box(np.array([self.action_angle_low]), np.array([self.action_angle_high]), dtype=np.float32)

        #--------------------- Observation space -----------------------
        self.obs_low_state = np.array([ 0, 0,        # wind_direction, wind speed (m/s)
                                        0,           # duration time
                                        0, 0,        # current position
                                       -1,           # last action
                                        0, 0, 0])    # last conc, current conc highest conc

        self.obs_high_state = np.array([2*math.pi, 20,
                                        self.max_step,
                                        self.court_lx, self.court_ly,
                                        1,
                                        self.conc_max, self.conc_max, self.conc_max])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        #--------------------------Ending Criteria--------------------------------
        self.conv_eps = 0.05
        self.eps = 1.0


    def init_envs(self, seed, pf_num, mean_on, gmm_num, kmeans_num, num_agents, max_agents, num_disc_actions=0):
        #----------------------- Random seed --------------------------
        self.seed_num = self.seed(seed)
        print("Seed: ", self.seed_num)

        #--------------------- user parameter number ---------------------------
        self.pf_num = pf_num # now set the same number of particle filter for all agents
        self.gmm_num = gmm_num
        self.kmeans_num = kmeans_num
        self.num_agents = num_agents
        self.max_agents = max_agents
        self.mean_on = mean_on
        self.num_disc_actions = num_disc_actions
        #------------------- Update observation ---------------------
        #----------------------- Mean input -------------------------
        if self.mean_on:
            mean_low = [0, 0, 0, 0, 0, 0] # mean_x, mean_y, mean_q, cov_x, cov_y, cov_q
            mean_high = [self.court_lx, self.court_ly, self.max_q,
                         self.court_lx**2, self.court_ly**2, self.max_q**2]
            self.obs_low_state = np.append(self.obs_low_state, mean_low)
            self.obs_high_state = np.append(self.obs_high_state, mean_high)
        #-------------------- GMM input state -----------------------
        if self.gmm_num > 0:
            for _ in range(self.gmm_num):
                self.obs_low_state = np.append(self.obs_low_state , [0, 0, 0, 0, 0]) # GMM: mean_x, mean_y, cov_x, cov_y, weight
                self.obs_high_state = np.append(self.obs_high_state, [self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2), 1])
        #------------------- KMeans input state -----------------------
        if self.kmeans_num > 0:
            for _ in range(self.kmeans_num):
                self.obs_low_state = np.append(self.obs_low_state , [0, 0, 0, 0]) # KMeans: mean_x, mean_y, cov_x, cov_y
                self.obs_high_state = np.append(self.obs_high_state, [self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2)])


        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        #-------------------- Init dummy gas condition -----------------------
        self.goal_x          = self.court_lx/2
        self.goal_y          = self.court_lx/2
        self.gas_d           = 10   # diffusivity [10m^2/s]
        self.gas_t           = 1000 # gas life time [1000sec]
        self.gas_q           = 2000 # gas strength
        self.wind_mean_phi   = 180  # mean wind direction
        self.wind_mean_speed = 2

    
    def _set_init_state(self): # initial state which is changed by env difficulty level
        # set initial state randomly
        self.goal_x = self.np_random.uniform(low=self.court_lx*0.2, high=self.court_lx*0.8)
        self.goal_y = self.np_random.uniform(low=self.court_ly*0.2, high=self.court_ly*0.8)
        self.gas_d = self.np_random.uniform(low=5, high=15)          # diffusivity [10m^2/s]
        self.gas_t = self.np_random.uniform(low=500, high=1500)      # gas life time [1000sec]
        self.gas_q = self.np_random.uniform(low=1000, high=3000)     # gas strength
        self.wind_mean_phi = self.np_random.uniform(low=0, high=360) # mean wind direction
        self.wind_mean_speed = self.np_random.uniform(low=0, high=4)

    def _dist_btw_points(self, p_1, p_2):
        return math.sqrt( (p_1[0]-p_2[0])**2 + (p_1[1]-p_2[1])**2 )

    def _conc_true(self, pos_x, pos_y): # true gas conectration
        if self.goal_x == pos_x and self.goal_y == pos_y: # to avoid divide by 0
            pos_x += 1e-10
            pos_y += 1e-10
        dist = math.sqrt(pow((self.goal_x - pos_x), 2) + pow(self.goal_y - pos_y, 2))
        y_n = -(pos_x - self.goal_x)*math.sin(self.wind_mean_phi*math.pi/180) \
              +(pos_y - self.goal_y)*math.cos(self.wind_mean_phi*math.pi/180)
        lambda_plume = math.sqrt(self.gas_d * self.gas_t / (1 + pow(self.wind_mean_speed,2) * self.gas_t/4/self.gas_d) )
        
        conc = self.gas_q/(4 * math.pi * self.gas_d * dist) * np.exp(-y_n * self.wind_mean_speed/(2*self.gas_d) - dist/lambda_plume)
        if conc < 0:
            conc = 0

        return conc #- self.conc_eps

    def _renew_reward(self, n):      # reward for the renew the highest gas conc
        #print(self.uav[0].gas_conc, " | ", self.uav[0].highest_conc)
        if self.uav[n].renew == True:
            reward = self.renew_reward #0.1 #+1
        else:
            reward = self.step_penalty #-1
        return reward

    def _crash_check(self, pre_pose_uav):
        vel = []
        root_data = np.zeros([self.num_agents, self.num_agents])
        for n in range(self.num_agents):
            vel.append([self.uav[n].x-pre_pose_uav[n][0],
                        self.uav[n].y-pre_pose_uav[n][1] ] )

        crash_list = np.zeros([self.num_agents, self.num_agents], dtype=bool)
        for n in range(self.num_agents):
            for m in range(n+1, self.num_agents):
                uav_m = self.uav[m]
                uav_n = self.uav[n]

                delta_p = np.array([uav_n.x-uav_m.x, uav_n.y-uav_m.y])
                delta_v = np.array([vel[n][0]-vel[m][0],
                                    vel[n][1]-vel[m][1] ])

                a_const = np.dot(delta_v, delta_v)
                b_const = 2*np.dot(delta_v, delta_p)
                c_const = np.dot(delta_p, delta_p) - self.crash_range**2
                if c_const< 0:
                    crash_list[n][m] = True
                    crash_list[m][n] = True
                    continue

                b_sq_4ac = b_const**2 - 4*a_const*c_const
                root_data[n][m] = b_sq_4ac
                if b_sq_4ac > 0:
                    root_1 = (-b_const + math.sqrt(b_sq_4ac) )/(2*a_const)
                    root_2 = (-b_const - math.sqrt(b_sq_4ac) )/(2*a_const)

                    if root_1 >= 0 and root_1 <= self.delta_t*4:
                        crash_list[n][m] = True
                        crash_list[m][n] = True
                        continue
                    elif root_2 >= 0 and root_2 <= self.delta_t*4:
                        crash_list[n][m] = True
                        crash_list[m][n] = True
                        continue

                #a_const = self.agent_v**2 * ( (math.cos(action[n]*math.pi)-math.cos(action[m]*math.pi))**2 +
                #                              (math.sin(action[n]*math.pi)-math.sin(action[m]*math.pi))**2 )
        #print("VELOCITY:")
        #print(vel)
        #print("ROOT_DATA:")
        #print(root_data)
        return crash_list


    def _comm_check(self):
        crash_warning_list = np.zeros([self.num_agents, self.num_agents], dtype=bool)

        self.comm_list  = np.zeros([self.num_agents, self.num_agents], dtype=bool)
        self.range_list  = np.zeros([self.num_agents, self.num_agents], dtype=bool)

        group = np.ones(self.num_agents, dtype=int)*(self.num_agents+1)
        for n in range(self.num_agents):
            if group[n] == (self.num_agents+1):
                temp = np.copy(group)
                temp[temp == (self.num_agents+1)] = 0
                group[n] = temp.max()+1
            for m in range(n+1, self.num_agents):
                uav_m = self.uav[m]
                uav_n = self.uav[n]
                m_n_dist = self._dist_btw_points([uav_m.x, uav_m.y], [uav_n.x, uav_n.y])
                if m_n_dist < self.crash_warning_range:
                    crash_warning_list[n][m] = True
                    crash_warning_list[m][n] = True
                if m_n_dist < self.wifi_range:
                    self.range_list[m][n] = True
                    if group[n] != group[m]:
                        max_group_indx = max(group[n], group[m])
                        min_group_indx = min(group[n], group[m])
                        group[n] = group[m] = min_group_indx
                        if max_group_indx <= self.num_agents:
                            for o in range(m):
                                if group[o] == max_group_indx:
                                    group[o] = min_group_indx
        for g_n in set(group):
            self.comm_list[group == g_n] = (group == g_n)
        #print(group)
                    
        for n in range(1, self.num_agents):
            for m in range(n):
                if self.comm_list[m][n] == False:
                    uav_m = self.uav[m]
                    uav_n = self.uav[n]
                    m_n_dist = self._dist_btw_points([uav_m.x, uav_m.y], [uav_n.x, uav_n.y])
                    if m_n_dist < self.wifi_range:
                        self.comm_list[m][n] = True
                        self.comm_list[n][m] = True

        return [crash_warning_list, group]



    def _communication_result(self, obs):
        comm_obs = copy.deepcopy(obs)

        for n in range(1, self.num_agents):
            for m in range(n):
                if self.comm_list[n][m]:
                    comm_obs[m] = np.append(comm_obs[m], obs[n])
                    comm_obs[n] = np.append(comm_obs[n], obs[m]) ###!! Just stack & no padding

        for n in range(self.num_agents):
            pad_size = (obs[n].size * self.max_agents) - comm_obs[n].size
            comm_obs[n] = np.pad(comm_obs[n], (0,pad_size), 'constant', constant_values=-1)


        return comm_obs
    
    ## ----------------------------- Each steps ----------------------------
    def step(self, action):
        self.count_actions += 1
        
        obs = []
        gas_concs = []
        wind_ds = []
        wind_ss = []

        pre_pose_uav = []
        for n in range(self.num_agents):
            pre_pose_uav.append([self.uav[n].x, self.uav[n].y])

        for n in range(self.num_agents):
            self.uav[n]._calculate_position(action[n])
            true_conc = self._conc_true(self.uav[n].x, self.uav[n].y)
            gas_concs.append(self.uav[n]._gas_measure(true_conc))
            wind_d, wind_s = self.uav[n]._wind_sensor(self.wind_mean_phi, self.wind_mean_speed) # Add error
            wind_ds.append(wind_d)
            wind_ss.append(wind_s)
        [crash_warning_list, group] = self._comm_check()
        crash_list = self._crash_check(pre_pose_uav)

        #print(group)

        for n in range(self.num_agents):
            for m in range(self.num_agents):
                if self.comm_list[m][n]:
                    self.uav[n]._estimator_update(gas_concs[m], self.uav[m].x, self.uav[m].y, wind_ds[m], wind_ss[m])
            single_obs = self.uav[n]._observation(gas_concs[n], wind_ds[n], wind_ss[n])

            obs.append( single_obs )

        ###-------------- commumication results -------------
        comm_obs = self._communication_result(obs)

        ###------------- About done of simulation -------------
        rews = np.zeros(self.num_agents)
        converges = []
        #nearby_bools = []
        for n in range(self.num_agents):
            converges.append(self.uav[n].cov_val)
        min_conv = np.min(converges)
        min_conv_indx = np.argmin(converges)

        converge_done = min_conv < self.conv_eps
        if min_conv < self.conv_eps:
            pf_center = self.uav[min_conv_indx].pf_center
            nearby = self._dist_btw_points(pf_center, [self.goal_x, self.goal_y])
            if nearby < self.eps:
                rews[group == group[min_conv_indx]] = self.success_reward
                #rews[min_conv_indx] += self.success_reward

        #-------------- Rewards for every steps -------------
        #print(crash_list)
        crash_done = False
        for n in range(self.num_agents):
            rews[n] += self._renew_reward(n) #When the agent renews highest conc, it get +0.1
            if np.sum(crash_list[n]) > 0:
                rews[n] += self.crash_penalty #When the agent makes collision, it get -100
                crash_done = True

            if np.sum(crash_warning_list[n]) > 0:
                rews[n] += self.crash_warning_penalty # Warning -0.1
            if np.sum(self.comm_list[n]) > self.num_agents/2: 
                rews[n] += self.comm_reward # When the agent is member of dominant group, it get +1


        timeout_done = bool(self.count_actions >= self.max_step)

        info = 1
        # There are three dones: Converge done, crash_done, timeout_done
        # print("CONV", converge_done, "CRASH", crash_done, "TIMEOUT", timeout_done)
        done = any([converge_done, crash_done, timeout_done])
        return comm_obs, rews, done, info

    def reset(self):
        print("Reset")
        self.count_actions = 0

        # set initial state randomly
        self._set_init_state()
        self.uav = []
        obs = []
        gas_concs = []
        wind_ds = []
        wind_ss = []
        ##------------------------------------- reset locations ----------------------------------------------
        uni_dist = self.np_random.uniform
        
        init_cent = uni_dist(low =[5, 5], high=[self.court_lx-5, self.court_ly-5])
        
        cent_to_source = math.sqrt( (self.goal_x-init_cent[0])**2 + (self.goal_y-init_cent[1])**2 )
        while cent_to_source < self.court_lx*0.3:
            init_cent = uni_dist(low =[5, 5], high=[self.court_lx-5, self.court_ly-5])
            cent_to_source = math.sqrt( (self.goal_x-init_cent[0])**2 + (self.goal_y-init_cent[1])**2 )
        
        init_source_dist = self.court_lx*0.3
        init_pos_low  = [max(init_cent[0] - self.crash_range * self.num_agents, 0),
                         max(init_cent[1] - self.crash_range * self.num_agents, 0)]
        init_pos_high = [min(init_cent[0] + self.crash_range * self.num_agents, self.court_lx),
                         min(init_cent[1] + self.crash_range * self.num_agents, self.court_ly)]

        for n in range(self.num_agents):
            init_pos = uni_dist(low =init_pos_low, high=init_pos_high)
            uav_to_source = self._dist_btw_points(init_pos, [self.goal_x, self.goal_y])
            if n==0:
                while uav_to_source < init_source_dist:
                    init_pos = uni_dist(low =init_pos_low, high=init_pos_high)
                    uav_to_source = self._dist_btw_points(init_pos, [self.goal_x, self.goal_y])
            else:
                for m in range(n):
                    uav_to_uav = self._dist_btw_points(init_pos, [self.uav[m].x, self.uav[m].y])
                    while (uav_to_source < init_source_dist or uav_to_uav < self.crash_range or uav_to_uav > self.wifi_range):
                        init_pos = uni_dist(low =init_pos_low, high=init_pos_high)
                        uav_to_source = self._dist_btw_points(init_pos, [self.goal_x, self.goal_y])
                        uav_to_uav = self._dist_btw_points(init_pos, [self.uav[m].x, self.uav[m].y])

            #init_pos = [10, 10]

            #self.normalization = False
            self.uav.append( Agent(self, init_pos, self.pf_num, self.mean_on, self.gmm_num, self.kmeans_num, self.normalization, self.num_disc_actions) )
            true_conc = self._conc_true(self.uav[n].x, self.uav[n].y)
            gas_concs.append(self.uav[n]._gas_measure(true_conc))
            wind_d, wind_s = self.uav[n]._wind_sensor(self.wind_mean_phi, self.wind_mean_speed) # Add error
            wind_ds.append(wind_d)
            wind_ss.append(wind_s)
        [_, _] = self._comm_check()


        for n in range(self.num_agents):
            for m in range(self.num_agents):
                if self.comm_list[m][n]:
                    self.uav[n]._estimator_update(gas_concs[m], self.uav[m].x, self.uav[m].y, wind_ds[m], wind_ss[m])
            self.uav[n].estimator.update_count = 0 # Forced to Zero
            single_obs = self.uav[n]._observation(gas_concs[n], wind_ds[n], wind_ss[n])
            obs.append( single_obs )

        comm_obs = self._communication_result(obs)
        #whole_obs = (np.array(obs).flatten()).tolist()

        return comm_obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
        
    ###=========================================== Graphical result rendering ===================================================            
    def render_background(self, mode='human'):
        size = self.screen_height / 500
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.background_viewer is None:
                self.background_viewer = rendering.Viewer(self.screen_width, self.screen_height)
            max_conc = 100;

            for xx in range(self.court_lx):
                for yy in range(self.court_ly):
                    conc = self._conc_true(xx+0.5, yy+0.5)
                    conc = self.np_random.normal(conc, math.sqrt(pow(self.env_sig,2) + pow(conc*self.sensor_sig_m,2)) )
                    while conc < 0:
                        conc = self.np_random.normal(conc, math.sqrt(pow(self.env_sig,2) + pow(conc*self.sensor_sig_m,2)) )

                    x = xx*self.scale
                    y = yy*self.scale
                    plume = rendering.make_circle(4.5*size)
                    plume.add_attr(rendering.Transform(translation=(x, y)))

                    if conc > max_conc: #set maximum value for visualization
                        conc = max_conc
                        color = cm.jet(255) # 255 is maximum number
                        plume.set_color(color[0], color[1], color[2])
                        self.background_viewer.add_onetime(plume)

                    elif conc > self.conc_eps + 0.5:
                        color_cal = round( (math.exp(math.log(conc+1)/math.log(max_conc+1))-1) * 255)
                        if color_cal < 0: color_cal = 0
                        color = cm.jet(color_cal)
                        plume.set_color(color[0], color[1], color[2])
                        self.background_viewer.add_onetime(plume)

            return self.background_viewer.render(return_rgb_array=mode == 'rgb_array')


    def render(self, mode='human'):
        size = self.screen_height / 500

        if mode == 'ansi':
            return self._observation()
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            ###-------------------------- Draw multiple agents ------------------------------
            for n in range(self.num_agents):
                track_way = rendering.make_polyline(np.dot(self.uav[n].positions, self.scale))
                track_way.set_linewidth(6 * size)
                self.viewer.add_onetime(track_way)

                # draw the agent
                agent = rendering.make_circle(10*size)
                agent.set_color(0, 0, 1)
                agent_trans = rendering.Transform()
                agent_trans.set_translation(self.uav[n].x * self.scale,
                                            self.uav[n].y * self.scale)
                agent.add_attr(agent_trans)
                self.viewer.add_onetime(agent)

                # draw measurement history
                for i in range(len(self.uav[n].conc_seq)):
                    measure = rendering.make_circle(math.pow(self.uav[n].conc_seq[i],1/3)*6*size)
                    measure.set_color(255, 0, 0)
                    measure_trans = rendering.Transform()
                    measure_trans.set_translation(self.uav[n].x_seq[i]*self.scale,
                                                  self.uav[n].y_seq[i]*self.scale)
                    measure.add_attr(measure_trans)
                    self.viewer.add_onetime(measure)

                if self.gmm_num > 0:
                    for i in range(self.uav[n].gmm_num):
                        for j in range(0, np.shape(self.uav[n].gmm_data[i])[0]):
                            particle = rendering.make_circle(3*size)
                            if i==0:
                                particle.set_color(0.8, 0.8, 0)
                            elif i==1:
                                particle.set_color(0.8, 0.1, 0.9)
                            elif i==2:
                                particle.set_color(1, 0.4, 0.1)
                            elif i==3:
                                particle.set_color(0.0, 0.8, 0.8)
                            elif i==4:
                                particle.set_color(0.1, 0.1, 0.9)
                            else:
                                particle.set_color(1, 0.4, 0.5)
                            particle_trans = rendering.Transform()
                            particle_trans.set_translation(self.uav[n].gmm_data[i][j][0]*self.scale,
                                                           self.uav[n].gmm_data[i][j][1]*self.scale)
                            particle.add_attr(particle_trans)                                
                            self.viewer.add_onetime(particle)

                        gmm_mean = rendering.make_capsule(19*size, 19*size)
                        gmm_mean.set_color(0,1,0)
                        gmm_mean_trans = rendering.Transform()
                        gmm_mean_trans.set_translation(self.uav[n].gmm_mean_x[i]*self.scale,
                                                       self.uav[n].gmm_mean_y[i]*self.scale)
                        gmm_mean.add_attr(gmm_mean_trans)
                        self.viewer.add_onetime(gmm_mean)
                elif self.kmeans_num > 0:
                    for i in range(self.uav[n].kmeans_num):
                        for j in range(0, np.shape(self.uav[n].km_data[i])[0]):
                            particle = rendering.make_circle(3*size)
                            if i==0:
                                particle.set_color(0.8, 0.8, 0)
                            elif i==1:
                                particle.set_color(0.8, 0.1, 0.9)
                            elif i==2:
                                particle.set_color(1, 0.4, 0.1)
                            elif i==3:
                                particle.set_color(0.0, 0.8, 0.8)
                            elif i==4:
                                particle.set_color(0.1, 0.1, 0.9)
                            else:
                                particle.set_color(1, 0.4, 0.5)
                            particle_trans = rendering.Transform()
                            particle_trans.set_translation(self.uav[n].km_data[i][j][0]*self.scale,
                                                           self.uav[n].km_data[i][j][1]*self.scale)
                            particle.add_attr(particle_trans)
                            self.viewer.add_onetime(particle)

                        km_mean = rendering.make_capsule(19*size, 19*size)
                        km_mean.set_color(0,1,0)
                        km_mean_trans = rendering.Transform()
                        km_mean_trans.set_translation(self.uav[n].km_mean_x[i]*self.scale,
                                                       self.uav[n].km_mean_y[i]*self.scale)
                        km_mean.add_attr(km_mean_trans)
                        self.viewer.add_onetime(km_mean)
                else:
                    for i in range(0,self.pf_num):
                        particle = rendering.make_circle(3*size)
                        particle.set_color(0,255,0)
                        particle_trans = rendering.Transform()
                        particle_trans.set_translation(self.uav[n].estimator.pf_x[i]*self.scale,
                                                       self.uav[n].estimator.pf_y[i]*self.scale)
                        particle.add_attr(particle_trans)
                        self.viewer.add_onetime(particle)

                if self.mean_on:
                    particles_center = rendering.make_capsule(19*size, 19*size)
                    particles_center.set_color(0.5, .3, .1)
                    particles_center_trans = rendering.Transform()
                    particles_center_trans.set_translation(self.uav[n].pf_center[0]*self.scale,
                                                           self.uav[n].pf_center[1]*self.scale)
                    particles_center.add_attr(particles_center_trans)
                    self.viewer.add_onetime(particles_center)


            goal = rendering.make_circle(10*size)
            goal.add_attr(rendering.Transform(translation=(self.goal_x*self.scale,
                                                           self.goal_y*self.scale)))
            goal.set_color(0, 0, 0)
            self.viewer.add_onetime(goal)

#            text = 'This is a test but it is not visible'
#            label = pyglet.text.Label(text, font_size=36,
#                                      x=10, y=10, anchor_x='left', anchor_y='bottom',
#                                      color=(255, 123, 255, 255))
#            label.draw()
#            self.viewer.add_geom(DrawText(label))

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.background_viewer:
            self.background_viewer.close()
            self.background_viewer = None



            

