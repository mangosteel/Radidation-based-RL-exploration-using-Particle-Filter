import math
import numpy as np
from gym_ste_v4.envs.common.particle_filter import ParticleFilter

import warnings
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')


class Agent:
    def __init__(self, env, init_pose, pf_num, mean_on, gmm_num, kmeans_num, obs_normalization, num_disc_actions):
        ### Initial state that declared 
        self.x = init_pose[0]
        self.y = init_pose[1]
        self.pf_num = pf_num
        self.mean_on = mean_on
        self.gmm_num = gmm_num
        self.kmeans_num = kmeans_num
        self.normalization = obs_normalization
        self.num_disc_actions = num_disc_actions

        ## True state that agent should know
        self.gas_d = env.gas_d
        self.gas_t = env.gas_t
        self.env_sig = env.env_sig
        self.sensor_sig_m = env.sensor_sig_m
        self.court_lx = env.court_lx
        self.court_ly = env.court_ly
        self.max_q = env.max_q
        self.agent_v = env.agent_v
        self.delta_t = env.delta_t
        self.max_step = env.max_step
        self.conc_max = env.conc_max
        self.seed_num = env.seed_num

        self.obs_low_state = env.obs_low_state
        self.obs_high_state = env.obs_high_state
        ### Initial state that should be reset for new episode
        self.dur_t = 0      # duration time of out of plume

        self.last_conc = 0
        self.highest_conc = 0
        self.step_rew = 0
        self.last_action = 0
        self.positions = []
        self.conc_seq = []
        self.x_seq = []
        self.y_seq = []

        ### Functions
        self.np_random = env.np_random
        self.estimator = ParticleFilter(self)
        self.pf_x = self.estimator.pf_x
        self.pf_y = self.estimator.pf_y
        self.pf_q = self.estimator.pf_q
        self.Wpnorms = self.estimator.Wpnorms

        seed_for_cluster = 1
        if self.gmm_num > 0:
            self.gmm = GaussianMixture(n_components=self.gmm_num, n_init=1, max_iter=20, random_state=seed_for_cluster)
            self.gmm_mean_x = np.ones(self.gmm_num)
            self.gmm_mean_y = np.ones(self.gmm_num)
            self.gmm_cov_x = np.ones(self.gmm_num)
            self.gmm_cov_y = np.ones(self.gmm_num)
            self.gmm_weights = np.ones(self.gmm_num)
        if self.kmeans_num > 0:
            self.kmeans = KMeans(n_clusters=self.kmeans_num, n_init=1, max_iter=20, random_state=seed_for_cluster)


    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, obs.size):
            normalized_obs.append((obs[i]-self.obs_low_state[i])/(self.obs_high_state[i] - self.obs_low_state[i]))
        return np.array(normalized_obs)


    def _wind_sensor(self, true_wind_phi, true_wind_speed):
        wind_degree_fluc = 5  #25 #degree
        wind_speed_fluc = 0.1 #1
        wind_dir = self.np_random.uniform(low=(true_wind_phi-wind_degree_fluc)/180,
                                         high=(true_wind_phi+wind_degree_fluc)/180)
        if wind_dir < 0:
            wind_dir += 2
        elif wind_dir > 2:
            wind_dir -= 2
        # wind_dir [radian]
        wind_speed = self.np_random.uniform(low=true_wind_speed-wind_speed_fluc, 
                                           high=true_wind_speed+wind_speed_fluc)
        return wind_dir*math.pi, wind_speed

    def _gas_measure(self, true_conc):
        env_sig = self.env_sig #1.0 #0.2 #0.4
        sensor_sig_m = self.sensor_sig_m #0.5 #0.1 #0.2

        conc_env = self.np_random.normal(true_conc, env_sig)

        while conc_env < 0: conc_env =0
            # conc_env = self.np_random.normal(true_conc, env_sig)
        gas_conc = self.np_random.normal(conc_env, conc_env*sensor_sig_m)
        while gas_conc < 0: gas_conc =0
            #gas_conc = self.np_random.normal(conc_env, conc_env*sensor_sig_m)

        return gas_conc

    def _estimator_update(self, gas_conc, x, y, wind_d, wind_s):
        self.estimator._weight_update(gas_conc, x, y,
                                      self.pf_x, self.pf_y, self.pf_q, self.Wpnorms,
                                      wind_d, wind_s)
        self.pf_x = self.estimator.pf_x
        self.pf_y = self.estimator.pf_y
        self.pf_q = self.estimator.pf_q
        self.Wpnorms = self.estimator.Wpnorms
        
    def _observation(self, gas_conc, wind_d, wind_s):
        #moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
        
        if gas_conc > self.highest_conc:
            self.highest_conc = gas_conc
            self.dur_t = 0
            self.renew = True
        else:
            self.dur_t += 1
            self.renew = False
        
#        self._estimator_update(gas_conc, self.x, self.y, wind_d, wind_s) # Already updated after comm check because self.comm_list[i][i]=1 always

        self.pf_x = self.estimator.pf_x
        self.pf_y = self.estimator.pf_y
        self.pf_q = self.estimator.pf_q
        self.Wpnorms = self.estimator.Wpnorms


        self.mean_x = sum(self.pf_x * self.Wpnorms)
        self.mean_y = sum(self.pf_y * self.Wpnorms)
        self.mean_q = sum(self.pf_q * self.Wpnorms)
        self.CovXxp = np.var(self.pf_x)
        self.CovXyp = np.var(self.pf_y)
        self.CovXqp = np.var(self.pf_q)
        
        self.pf_center = [self.mean_x, self.mean_y]

        obs = np.array([float(wind_d), float(wind_s),
                        float(self.dur_t),
                        float(self.x), float(self.y),
                        float(self.last_action),
                        float(self.last_conc), float(gas_conc), float(self.highest_conc)])

        if self.mean_on:
            obs_mean =  [float(self.mean_x), float(self.mean_y), float(self.mean_q),
                         float(self.CovXxp), float(self.CovXyp), float(self.CovXqp)]

            obs=np.append(obs, obs_mean)

        if self.gmm_num > 0:
            if self.estimator.update_count == 0: # only update after resampling
                pf_con = np.column_stack((self.pf_x, self.pf_y))
                gmm_labels = self.gmm.fit_predict(pf_con)
                self.gmm_weights = self.gmm.weights_
            
                self.gmm_data = []
                for k in range(self.gmm_num):
                    self.gmm_data.append(pf_con[gmm_labels == k])
                    gmm_Wpnorms = self.Wpnorms[gmm_labels == k]
                    gmm_Wpnorms = gmm_Wpnorms/sum(gmm_Wpnorms)
                    data_split = np.transpose(self.gmm_data[k])

                    self.gmm_mean_x[k] = sum(data_split[0] * gmm_Wpnorms)
                    self.gmm_mean_y[k] = sum(data_split[1] * gmm_Wpnorms)
                    if np.shape(data_split[0])[0] == 0:
                        self.gmm_cov_x[k] = 0
                        self.gmm_cov_y[k] = 0
                    else:
                        self.gmm_cov_x[k] = np.var(data_split[0])
                        self.gmm_cov_y[k] = np.var(data_split[1])

            for k in range(self.gmm_num):
                obs = np.append(obs, [float(self.gmm_mean_x[k]), float(self.gmm_mean_y[k]), float(self.gmm_cov_x[k]), float(self.gmm_cov_y[k]), float(self.gmm_weights[k])])

        if self.kmeans_num > 0:
            if self.estimator.update_count == 0: # only update after resampling
                pf_con = np.column_stack((self.pf_x, self.pf_y))
                kmeans_labels = self.kmeans.fit_predict(pf_con)

                self.km_mean_x = np.ones(self.kmeans_num)
                self.km_mean_y = np.ones(self.kmeans_num)
                self.km_cov_x = np.ones(self.kmeans_num)
                self.km_cov_y = np.ones(self.kmeans_num)
                self.km_data = []
                for k in range(self.kmeans_num):
                    self.km_data.append(pf_con[kmeans_labels == k])
                    km_Wpnorms = self.Wpnorms[kmeans_labels == k]
                    km_Wpnorms = km_Wpnorms/sum(km_Wpnorms)
                    data_split = np.transpose(self.km_data[k])

                    self.km_mean_x[k] = sum(data_split[0] * km_Wpnorms)
                    self.km_mean_y[k] = sum(data_split[1] * km_Wpnorms)
                    if np.shape(data_split[0])[0] == 0:
                        self.km_cov_x[k] = 0
                        self.km_cov_y[k] = 0
                    else:
                        self.km_cov_x[k] = np.var(data_split[0])
                        self.km_cov_y[k] = np.var(data_split[1])

            for k in range(self.kmeans_num):
                obs = np.append(obs, [float(self.km_mean_x[k]), float(self.km_mean_y[k]), float(self.km_cov_x[k]), float(self.km_cov_y[k])] )


        self.last_conc = gas_conc
        
        self.conc_seq.append(gas_conc)
        self.x_seq.append(self.x)
        self.y_seq.append(self.y)
        self.positions.append([self.x, self.y])
        self.cov_val = np.sqrt(self.CovXxp/pow(self.court_lx,2) + \
                               self.CovXyp/pow(self.court_ly,2) + \
                               self.CovXqp/pow(self.max_q,2) )

        if self.normalization:
            obs = self._normalize_observation(obs)

        return obs
        
    def _calculate_position(self, action):
        if self.num_disc_actions > 0:
            disc_action = np.argmax(action)/self.num_disc_actions
            angle = (disc_action) * math.pi 
            self.last_action = disc_action
        else:
            angle = (action) * math.pi
            self.last_action = action
        step_size = self.agent_v * self.delta_t
        # calculate new agent state
        self.x = self.x + math.cos(angle) * step_size
        self.y = self.y + math.sin(angle) * step_size

        # borders
        if self.x < 0:
            self.x = 0
        if self.x > self.court_lx:
            self.x = self.court_lx
        if self.y < 0:
            self.y = 0
        if self.y > self.court_ly:
            self.y = self.court_ly

