import sys         # 3/19 목표 : 코드 분석완료... // 3/19 : rendering 부분 빼고는 분석완료... / 그러면 3/20에 rendering에서 source 관련부분 있는지 확인하자..
import copy        # 3/20 : 가스 모델 시각화 불러오기 성공! / 방사능 모델도 시각화 성공(경로꼬인것이 원인/)
import gym         # 3/21 : 어떤 부분을 수정해야 하는지 , 가스 >> 방사선(어차피 변수랑 함수부분만 수정하면됨...) / 일단은 추가만 해놓고 가스 주석처리함면서 한꺼번에 다 바꾸자!
from gym import error, spaces, utils   # 3/21 : 방사능 모델로 수정했고 일단 돌아감!
from gym.utils import seeding
import numpy as np
import pyglet
from matplotlib import cm

from gym_ste_v4.envs.common.utils import *
from gym_ste_v4.envs.multi_rotary_wing.agent import Agent

from gym.envs.registration import register
from datetime import datetime
import time
import math # 수정
class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self): # necessary to draw somthing
        self.label.draw()

class GasClass():  # 가스 객체 >> 방사선 객체로 수정해야함!
    def __init__(self):
        self.S_x = None  # 가스의 좌표 >> 방사선의 누출 좌표
        self.S_y = None
        self.d   = None  # 확산계수 (diffusivity) : 가스가 퍼지는 속도? 이므로 이미 활동도는 q로 대체가 되는상태라 없어도 될듯...
        self.t   = None  # 방사선은 정적상태를 가정하기 때문에 없어도 됨!
        self.q   = None  # q >>> x(방출강도) b = Ax!

class Radiation():  # 방사선 객체 (1)
    def __init__(self):
        self.S_x = None  # 방사선의 누출 좌표 / 애초에 radi_x에 해당하는 활동도는 cs-137에서 고정되어 있기에 애초에 범위가 필요없음!!
        self.S_y = None
        self.S_dose_rate = None # 이제 이것을 방출강도와 같은 포지션으로 보고 각 파티클에 대해서 다 다르게 해보자... 단 수치값을 현실적으로..
        self.S_mass = None # 최대값의 70퍼센트를 초기질량으로 둠! 
        self.cs_137_gamma = 330/(1000) # nSV*m^2/(h*kBq) / 이제 시간단위는 h로 고정!
        self.special_activity = 3.2*10**12 # Bq/g 이건 고정!
        

class TrueWind():  # 바람객체  >> 고려안해도 됨!
    def __init__(self):
        self.mean_phi   = None  # True wind mean direction
        self.mean_speed = None

class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        self.normalization = True # 정규화 여부
    
        #---------------------- important settings ---------------------
        self.test = False
        self.env_sig = 0.2 #  방사선도 바람은 아니지만 공기산란의 영향을 받는다고 보고 그대로 노이즈는 가져가자...            
        self.sensor_sig_m = 0.1 
        self.agent_v = 4                # 2m/s 원래 4인데 2로 줄여봄...
        self.delta_t = 1                # 1sec
        self.comm_range = 10    # 10m 
        self.crash_warning_range = 8
        self.block_on = False
        self.step_size = self.agent_v * self.delta_t # 한번 이동할때 거리...
        self.crash_range = self.step_size / 2   
        self.motion_noise_std = 0.1

        self.success_reward        = +100  # 근원지 찾았을때(거리가 거의0에 가까울때...)
        self.crash_warning_penalty = +0 #-0.1
        self.crash_penalty         = +0 #-100 or 0 for disable
        self.comm_reward           = +0.01 #+0.01 
        self.step_penalty          = -1.0 # 기본적으로 step을 수행할때 음의 보상을 받고 만약 최선의 측정값을 못찾으면 보상이 빠르게 감소함.
        self.renew_reward          = +0.1 # 새로운 최고 가스 농도(highest conc)를 갱신할 때 부여되는 보상
        #--------------------------Common Env----------------------------
        self.debug = False
        self.nearby = None
        self.last_measure = 0 # 이전 측정값을 말하는건가? 아마도 시작전에는 농도가 0아닐까.. 감지자체가 안되니...
        self.court_lx = 60              # the size of the environment
        self.court_ly = 60              # the size of the environment
        self.max_step = 150  # 최대 150스텝으로 에피소드를 정해놓는 건가?

        # radiation sensing
        # self.conc_eps = 0.2 # minimum conc  # 가스 센싱의 범위를 정의 해놓은거 같다.
        # self.conc_max = 100
        self.max_mass = 9.47 * 10**(-6)
        self.min_mass = (9.47 * 10**(-6)) /2
        self.dose_eps =  (9.47 * 10**(-6)/2)*  3.2*10**12 /(60)**(2) *(330/1000)  #  does rate 범위 설정! (2) 
        self.dose_max =  (9.47 * 10**(-6))*  3.2*10**12 /(1)**(2) *(330/1000)  # 내 생각엔 이 수치를 먼저 바꿔야 될거 같다.. 어차피 그러면 나머지는 비슷할듯??>  # r=0.1일때를 최대 선량률로 본다 .그리고 거리가 가장멀때는 60m이므로 r=60을 가정함!
        
        # rendering                                                 #그리고 단위를 h>>s로 변환함으로써 조금더 현실성있는 값으로 변환함!
        self.screen_height = 400                                 # dose_rate : 0.184 ~ 66586.6
        self.screen_width = 400
        self.viewer = None                  # viewer for render()
        self.background_viewer = None       # viewer for background
        self.background_viewer_radi = None
        self.scale = self.screen_width/self.court_lx
        self.true_dose_rate = np.zeros((self.court_lx, self.court_ly)) # 아마 비디오 화면에 농도값을 넣어서 표시할려나? 그건듯? / 이건 그냥 쓰자 굳이 렌더링 변수라굳이

        #--------------------------Initial data-------------------------
        self.gas_measure = -1 # 실제 가스농도 측정값 >> does rate 실제 측정값
        
        self.dose_measure = -1  # >> does rate 실제 측정값 (3)

        self.outborder = False          # in search area

        self.update_count = 0

        self.warning = False 

        # self.max_q = 5000 # q >> x /  근데 어차피 이건 값 자체에 제한 두는 거라 미리 설정하느 거네..
        # self.max_radi_x = 5000 # 이건 완전 잘못된 거임! 애초에 cs-137은 활동도가 고정....

        # 초반에 이거 수정할때 실수한게 너무 가스 물리량에 의존했다는 거임!... 

        #------------------------ Action space ------------------------
        self.action_angle_low  = -1 # 액션값은 각도가 나와야 하는데 아마도 pi를 곱해서 -180 ~180이겠지?
        self.action_angle_high =  1
        self.action_space = spaces.Box(np.array([self.action_angle_low]), np.array([self.action_angle_high]), dtype=np.float32)
        # 액션이 [-1,1]의 범위를 가지는 연속적인 공간에서 정의된다. step할때 액션 뽑을때 사용될듯??
        
        #--------------------- Observation space -----------------------  # 아마 이 obs 공간은 gmm클러스터링을 제외한 관측데이터만 포함하는듯...
        # self.obs_low_state = np.array([ 0, 0,        # wind_direction, wind speed (m/s) / 바람 정보고려안함! 
        #                                 0,           # duration time  / 고려안해도됨!/ 정적상태를 가정함!/ 그 최고농도 찾을떄까지걸린 스텝을 의미하나? 그러면 넣는게 ㅇㅇ
        #                                 0, 0,        # current position of sensor / 뭐 사실상 에이전트 위치네..
        #                                -1,           # last action  
        #                                 0, 0, 0])    # last conc, current conc highest conc >> radiation does rate : 방사선 선량률

        # self.obs_high_state = np.array([2*math.pi, 20,  # 그냥 관측값의 최대값을 관측 공간에 정의해 놓은 것.
        #                                 self.max_step,  
        #                                 self.court_lx, self.court_ly,
        #                                 1,
        #                                 self.conc_max, self.conc_max, self.conc_max]) # >> radiation does rate : 방사선 선량률
        

        self.obs_low_state = np.array([ 0,           # duration time   (4)
                                        0, 0,       # current position of sensor / 뭐 사실상 에이전트 위치네..
                                       -1,          # last action  
                                        0, 0, 0])   # last does, current does,  highest dose >> radiation does rate : 방사선 선량률

        self.obs_high_state = np.array([self.max_step,   # (5)
                                        self.court_lx, self.court_ly,
                                        1,
                                        self.dose_max, self.dose_max, self.dose_max]) # >> radiation does rate : 방사선 선량률


        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32) # 이런식으로 공간을 정의하는 구나....

        #--------------------------Ending Criteria--------------------------------
        self.conv_eps = 0.05  # 아마 수렴하는 기준인듯? 거리같은거? / 3.19 완료...
        self.eps = 4.0


    def init_envs(self, seed, adj_act_on, crash_check_on, pf_num, mean_on, lidar_on,gmm_num, kmeans_num, num_agents, num_disc_actions=0):
        #----------------------- Random seed --------------------------
        self.seed_num = self.seed(seed)  # 
        print("Seed: ", self.seed_num)

        #--------------------- user parameter number ---------------------------
        self.pf_num = pf_num # now set the same number of particle filter for all agents
        self.gmm_num = gmm_num
        self.kmeans_num = kmeans_num
        self.lidar_on = lidar_on
        self.num_agents = num_agents # 어차피 난 1개로 할꺼임!
        self.mean_on = mean_on  # 아마도 관측(상태) 데이터에 gmm클러스터링 데이터를 추가 여부 판단 불리언
        self.num_disc_actions = num_disc_actions # 이상행동 개수인데 이게 뭘 의미하는걸까? 나중에 보자
        self.adjusted_action_on = adj_act_on # default = False
        self.crash_check_on = crash_check_on
        self.num_beam = 10
        

        #------------------- Update observation --------------------- # 아마도 gmm클러스터링 데이터(agent에서 계산된)를 관측정보에 추가하는 내용이 들어갈거 같다.
        #----------------------- Mean input -------------------------
        if self.mean_on:
            mean_low = [0, 0, self.min_mass, 0 ,0 ,self.min_mass**2]  # mean_x, mean_y, mean_mass, cov_x, cov_y, cov_mass
            mean_high = [self.court_lx, self.court_ly, self.max_mass,  
                         self.court_lx**2, self.court_ly**2,self.max_mass**2] # max.q >> max.S_dose_rate (6) 근데 뭐 범위값은 사실상 같긴해 그냥 이름만...
            self.obs_low_state = np.append(self.obs_low_state, mean_low)
            self.obs_high_state = np.append(self.obs_high_state, mean_high)
            # 어차피 파티클 평균은 한번만 계산되니깐 5개추가!
        #-------------------- GMM input state -----------------------
        if self.gmm_num > 0:
            for _ in range(self.gmm_num):
                self.obs_low_state = np.append(self.obs_low_state , [0, 0, 0, 0, 0]) # GMM: mean_x, mean_y, cov_x, cov_y, weight
                # 여기서 클러스터 1개당 mean_x, mean_y, cov_x, cov_y, weight가 들어가므로 3개면 15개의 요소가 추가됨...
                self.obs_high_state = np.append(self.obs_high_state, [self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2), 1])
        #------------------- KMeans input state -----------------------
        if self.kmeans_num > 0:
            for _ in range(self.kmeans_num):
                self.obs_low_state = np.append(self.obs_low_state , [0, 0, 0, 0]) # KMeans: mean_x, mean_y, cov_x, cov_y
                self.obs_high_state = np.append(self.obs_high_state, [self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2)])



        if self.lidar_on: # 빔개수는 10개지만 사실 180도를 10부분으로 나눈것을 의미함! 경계기준 같은거임! 
            lidar_low = [0] * self.num_beam
            lidar_high = [self.court_lx] * self.num_beam # 10차원 만 추가...
            self.obs_low_state = np.append(self.obs_low_state, lidar_low)
            self.obs_high_state = np.append(self.obs_high_state, lidar_high)
            
        
        print(f"state_size : {self.obs_low_state.size}")
        print()


        self.observation_space = spaces.Box(np.tile(self.obs_low_state,  self.num_agents), #tile은 에이전트 개수만큼 복사하는 것. 
                                            np.tile(self.obs_high_state, self.num_agents), dtype=np.float32)

        #-------------------- Init dummy radiation condition -----------------------
        self.gas       = GasClass()  #  가스객체. >> 방사선 객체로 바꾸면 댐!
        self.gas.S_x   = self.court_lx/2  # 초기위치는 전체영역의 절반에 위치..
        self.gas.S_y   = self.court_lx/2
        self.gas.d     = 10   # diffusivity [10m^2/s] / 고려안해도됨!
        self.gas.t     = 1000 # radiation life time [1000sec] / 지속시간? 스텝당 업데이트되는 시간과는 다른거 인듯? /duration time이랑 다른듯 / 고려안해도됨!
        self.gas.q     = 2000 # radiation strength
        self.t_wind            = TrueWind() # 바람정보 고려안함!
        self.t_wind.mean_phi   = 180  # True wind mean direction 
        self.t_wind.mean_speed = 2

        #-------------------- Init dummy radiation condition -----------------------  (7)
        self.radiation       = Radiation()  #  가스객체. >> 방사선 객체로 바꾸면 댐!
        self.radiation.S_x   = self.court_lx/2  # 초기위치는 전체영역의 절반에 위치..
        self.radiation.S_y   = self.court_lx/2
        self.radiation.S_mass = 9.47 * 10**(-6) * 0.7 # 아마도 1m기준으로 10mSV/h 인가 아마도?


    
    def _set_init_state(self): None # initial state which is changed by env difficulty level
        

    def _renew_reward(self, n):      # reward for the renew the highest radiation conc
        #print(self.uav[0].gas_conc, " | ", self.uav[0].highest_conc)
        if self.uav[n].renew == True: # 우선 reset메서드를 먼저 참고하고 오자! / 알고보니 uav[n]은 Agent객체이다!
            reward = self.renew_reward # 0.1  # +1 / 가스최고 농도를 측정했을때 0.1받음!
        else:
            reward = self.step_penalty #-1
        return reward

    ## ----------------------------- Each steps -----------------------------
    def step(self, action):   
        self.count_actions += 1 # step함수 호출시 카운트를 올린다.
        self.action_for_cast = action
        
        global_obs = [] #  reset에서 정의한거랑 이름은 같은데 함수안이라 겹치지는 않을듯?
        radiation_dose_rate = []  # 이게 아마 센서 측정 농도를 담는 리스트... >> does rate 로 .. 그냥 그대로 쓰자 솔직히 이름만 바꾸는 건데..
        # 변수명 한번에 바꾸기 : ctrl +shift + L !

        # wind_ds = []  # 풍향 / 바람고려안함!
        # wind_ss = []  # 풍속

        pre_pose_uav = [] # 에이전트의 초기위치를 말하는듯...
        for n in range(self.num_agents):
            pre_pose_uav.append([self.uav[n].x, self.uav[n].y])


        

        for n in range(self.num_agents): # it is necessary in base.py
            pre_pose_temp = [row for i, row in enumerate(pre_pose_uav) if i != n] # 여기서 uav는 agent객체임!
            #print("Agent ", n)
            self.uav[n]._calculate_position(action[n], pre_pose_temp) # 근데 1대만 있다면 빈 리스트 [] 나올듯?
            # true_dose_rate = isotropic_plume(self.uav[n].x, self.uav[n].y, self.radiation, self.t_wind) # 노이즈 없는 파티클에서의 평균 가스농도 >> utils에서 방사선 선량 계산함수 하나 만들면됨!
            
            true_dose = radiation_field(self.uav[n].x, self.uav[n].y, self.radiation,visual=False, obstacles = self.obstacles) # dose계산시 obstacle 정보 반드시 주기!!

            radiation_dose_rate.append(self.uav[n]._radiation_measure(true_dose)) # 실제 센서에서 측정한 가스농도값 >> 실제 노이즈있는 방사선 선량률...
            # wind_d, wind_s = self.uav[n]._wind_sensor(self.t_wind.mean_phi, self.t_wind.mean_speed) # Add error >> 고려할 필요없음!
            # wind_ds.append(wind_d) # >> 고려할 필요없음!
            # wind_ss.append(wind_s) # reset에서 했던 작업의 반복임! >> 고려할 필요없음!


        # _, self.dist = self.simulate_lidar(self.uav[0].x,self.uav[0].y, action[0]*math.pi)
        # self.uav[0].lidar_val = self.dist
        
        # x0, y0 = self.uav[0].last_x, self.uav[0].last_y
        # x1, y1 = self.uav[0].x, self.uav[0].y
        
        
        # if min(self.dist) < 0.5 or self.path_intersects_obstacle(x0, y0, x1, y1, self.obstacles):
        #     self.block_on = True
        #     print(f"crack!!!  cross : {self.path_intersects_obstacle(x0, y0, x1, y1, self.obstacles)}")
        #     print()

        [group, multi_hop_matrix, laplacian_matrix] = comm_check(self.num_agents, self.uav, self.comm_range)
        # Check connection group(including Multi-hop), Multi-hop Matrix, Single-hop Matrix
        if self.crash_check_on:
            [crash_list, crash_warning_list] = crash_check(pre_pose_uav, self.num_agents,
                                                               self.uav, self.crash_range, self.crash_warning_range,
                                                               self.step_size, self.delta_t)
        # Check whether any agent crashed between "current time step" and "previous time step" in "crash_check()"

        #print(group)

        for n in range(self.num_agents):
            num_connected = sum(laplacian_matrix[n])
            for m in range(self.num_agents):
                if laplacian_matrix[m][n]: # 이제는 agent 수정하러 가자~~ (9) _estimator_update() / _observation()
                    self.uav[n]._estimator_update(radiation_dose_rate[m], self.uav[m].x, self.uav[m].y, num_connected) #바람정보 제외!
                    # ._estimator_update()에서 weight_update()하고 pf_x,y,q는 리샘플링>스무딩을 거쳐서 다시 갱신된다..
            single_obs = self.uav[n]._observation(radiation_dose_rate[n])  # 바람 정보 제외!

            global_obs.extend( single_obs.tolist() ) # 리턴될 관측 데이터!

        ###------------ commumication results ---------------
        comm_obs = decentralized_obs_for_each_agent(np.array(global_obs), self.num_agents, laplacian_matrix)
        # 어? comm_obs 에도 global_obs가 들어가네? 에이전트 1개면 동일하게 되나?
        ###----------- About done of simulation -------------
        rews = np.zeros(self.num_agents) # 리워드 / 1개(에이전트 1대)
        converges = []
        #nearby_bools = []
        for n in range(self.num_agents): # agent 251라인
            converges.append(self.uav[n].cov_val) # , _observation()에서 계산됨! , 파티클들이 서로 얼마나 모여 있는지를 정량화한 값임!
        min_conv = np.min(converges) # 애초에 pf_x,y의 분산이 들어가므로 , 파티클들이 서로 모여있으면 분산이 작아져서 값이 작아짐.
        min_conv_indx = np.argmin(converges)  # 

        converge_done = min_conv < self.conv_eps # 임계치보다 작다면.. done이 true가 되겠지.. 

        # if self.block_on:
        #     self.crash = 100
        #     rews[group == group[min_conv_indx]] = - self.crash
        
        if min_conv < self.conv_eps: # 이것이 실행된다는 건 일단 true인 상태...
            pf_center = self.uav[min_conv_indx].pf_center # 어차피 에이전트 1개면 argmin이 의미없긴함..
            self.nearby = d_btw_points(pf_center, [self.radiation.S_x, self.radiation.S_y]) 
            
            if self.nearby < self.eps: # 거리가 임계치보다 작으면, 보상값 100을 받는다.
                rews[group == group[min_conv_indx]] = self.success_reward * sum(group == group[min_conv_indx])/self.num_agents
                
           
                #rews[min_conv_indx] += self.success_reward

        #-------------- Rewards for every steps -------------
        #print(crash_list)
        crash_done = False

        for n in range(self.num_agents):
            rews[n] += self._renew_reward(n) #When the agent renews highest conc, it get +0.1
            if self.crash_check_on:
                if np.sum(crash_list[n]) > 0:
                    rews[n] += self.crash_penalty #When the agent makes collision, it get -100
                    crash_done = True
                if np.sum(crash_warning_list[n]) > 0:
                    rews[n] += self.crash_warning_penalty # Warning -0.1
            if np.sum(laplacian_matrix[n]) > self.num_agents/2: 
                rews[n] += self.comm_reward # When the agent is member of dominant group, it get +1


        timeout_done = bool(self.count_actions >= self.max_step)
        
        if not converge_done:
            self.nearby = None  # 그래서 수렴할때와 발산할 때의 nearby가 같구나... 애초에 거리조건만 만족안시키는 경우라서 거리가 계산된거 구나?

        info = [converge_done, crash_done, timeout_done]
    

        # There are three dones: Converge done, crash_done, timeout_done
        # print("CONV", converge_done, "CRASH", crash_done, "TIMEOUT", timeout_done)
        done = any([converge_done, crash_done,timeout_done])
        # self.uav[0].last_x , self.uav[0].last_y = self.uav[0].x , self.uav[0].y
        return [global_obs, comm_obs], rews, done, info # 아마 에이전트를 설정할때, 파티클정보야 원래 균일분포로 초기화되는건데...
                                                        # 바람정보나 센서측정값을 기반으로 관측 데이터가 달라질듯?(wegiht_update, gmm 클러스터링...) 

    def reset(self): # 그럼 리셋을 먼저 한다는  가정하에 uav가 미리 정의되고 init_env에서 self.uav를 사용할수 있는듯? 어차피 호출먼저 되는 장떙이니...
        print("Reset")
        self.count_actions = 0
        # self.block_on = False

        # set initial state randomly
        self._set_init_state() # 아 그러면 env.step시 extreme에서 정의한 최종 환경객체의 set_init_state()가 여기서 발동되는거네...!
        
        if self.test:
            print(self.test)
            self.radiation.S_x = self.np_random.uniform(low=self.court_lx*0.85, high=self.court_lx*0.95)  # 그냥 이렇게 따로 하드코딩 해야될듯? test에서는 따로 업데이트가 안되는듯??
            self.radiation.S_y = self.np_random.uniform(low=self.court_lx*0.85, high=self.court_lx*0.95)  # 이건 테스트 전용! 다행히 잘 찾아감! # 흥미로운 점은 no_pf는 아예 못찾음.. 절반 고정은 잘찾는데...오호.. 
        
        self.uav = []
        global_obs = []
        radiation_dose_rate = []
        # wind_ds = []
        # wind_ss = []
        # init_pos = np.random.uniform(low=[5,5],high=[30,30], size=[num_agents,2]) / (5,5) ~(30,30) 사이로 초기위치를 정함!
        
        init_pos = set_init_pos(self.np_random, self.court_lx, self.court_ly, self.radiation, self.crash_range, self.num_agents, self.comm_range)

        for n in range(self.num_agents):
            self.uav.append( Agent(self, init_pos[n], self.pf_num, self.mean_on, self.gmm_num, self.lidar_on,self.kmeans_num, self.normalization, self.num_disc_actions) )
            # 이것이 바로 uav[n]의 실체! >> uav[n] : Agent....
            self.obstacles = self.generate_maze_obstacles(agent_start=(self.radiation.S_x,self.radiation.S_y))


            # true_dose_rate = isotropic_plume(self.uav[n].x, self.uav[n].y, self.radiation, self.t_wind) # conc가 노이즈 없는 평균 가스 농도/ 논문에서는 m(p|seta)로 표기함..
            true_dose_rate = radiation_field(self.uav[n].x, self.uav[n].y, self.radiation,visual=False, obstacles = self.obstacles) # 근데 여기에 장애물을 없어도 되나?
            # isotropic_plume >> radiation_field로 교체!
            radiation_dose_rate.append(self.uav[n]._radiation_measure(true_dose_rate)) # For update all agent using other agents' measurement
            # agent._radiation_measure()는 단순히/ 평균 가스농도랑 바람,센서노이즈를 평균, 분산으로 가지는 가우시안 분포에서 뽑은 측정값임! 

            # wind_d, wind_s = self.uav[n]._wind_sensor(self.t_wind.mean_phi, self.t_wind.mean_speed) # Add error / 이건 제외해도 됨!

            # 풍향과 풍속에 약간의 노이즈를 의도적으로 첨가하는 역할을 함!
            # wind_ds.append(wind_d) # 이건 제외해도 됨!
            # wind_ss.append(wind_s) # 이건 제외해도 됨!

        [group, multi_hop_matrix, laplacian_matrix] = comm_check(self.num_agents, self.uav, self.comm_range) # 이건 나중에 보던가...


        for n in range(self.num_agents):
            num_connected = sum(laplacian_matrix[n])
            for m in range(self.num_agents):
                if laplacian_matrix[m][n]: # 근데 파티클 정보가 없는데? 
                    self.uav[n]._estimator_update(radiation_dose_rate[m], self.uav[m].x, self.uav[m].y, num_connected) # 바람제외!
                    # 여기서 _estimator_update()는 wegiht_update()를 하는데, 이때 particle 정보가 필요하다. 이것은 파티클 필터의 pf값을 agent에서 한번더 덮어쓴거라서 사용가능한 것이다.
                    #  pf,x,y의 범위는 0~court_x,y이고, q는 0~max_q, wegiht는 1/n 으로 초기화되어 있다.
            self.uav[n].estimator.update_count = 0 # Forced to Zero if / self.estimator.update_count == 0: 이거 gmm클러스터링 조건임! 그래서 0으로 맞추는거!
            single_obs = self.uav[n]._observation(radiation_dose_rate[n]) # _observation()호출해서 Agent.renew가 호출된다! / 바람정보제외!
            # 이때 observation은 기존의 관측값 바람,센서값, 초기위치등의 데이터 + MVG(gmm 클러스터링 데이터) 합친 데이터가 정규화되서 나온다!
            global_obs.extend( single_obs.tolist() ) # staking all single_obs / 아 원래는 멀티 에이전트면 각 에이전트의 obs를 다 리스트에 확장시키는건데 1개라서 하나의 리스트만 나올듯!

        comm_obs = decentralized_obs_for_each_agent(np.array(global_obs), self.num_agents, laplacian_matrix)
        #whole_obs = (np.array(obs).flatten()).tolist()
        # self.obstacles = self.generate_maze_obstacles(agent_start=(self.radiation.S_x,self.radiation.S_y))

        return [global_obs, comm_obs] # 그래서 이게 env.reset()하면 나오는 상태 데이터 구나!

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
    

    def path_intersects_obstacle(self,x0, y0, x1, y1, obstacles):
        for (p1, p2, *_) in obstacles:
            if line_intersects(x0, y0, x1, y1, p1[0], p1[1], p2[0], p2[1]):
                return True
        return False
    
    def simulate_lidar(self, x, y, heading, num_beams=180, fov=math.pi,sensor_range=10.0, noise_std=0.05):
        """
        LiDAR 시뮬레이션 (2D):
        - x, y: 에이전트 위치
        - heading: 에이전트 회전 각도 (라디안)
        - num_beams: 레이 개수
        - fov: 시야각 (라디안)
        반환: 거리 리스트
        """
        angles = [ heading - fov / 2 + (fov * i) / (num_beams - 1) for i in range(num_beams)]
        
        distances = []
        for theta in angles:
            dx = math.cos(theta)
            dy = math.sin(theta)
            max_dist = sensor_range
            x2 = x + dx * max_dist
            y2 = y + dy * max_dist
            min_dist = max_dist
            for obs in self.obstacles:
                (x1, y1), (x3, y3), *_ = obs
                denom = (x3 - x1)*(y - y2) - (y3 - y1)*(x - x2)
                if abs(denom) < 1e-8:
                    continue
                t = ((x - x1)*(y - y2) - (y - y1)*(x - x2)) / denom
                u = ((x3 - x1)*(y - y1) - (y3 - y1)*(x - x1)) / denom
                if 0 <= t <= 1 and u >= 0: # 확인!
                    
                    ix = x + u * (x2 - x)
                    iy = y + u * (y2 - y)
                    dist = math.hypot(ix - x, iy - y)
                    if dist < min_dist:
                        min_dist = dist
            noisy_dist = min_dist + np.random.normal(0, noise_std)
            noisy_dist = max(0.0, min(noisy_dist, sensor_range))  # 범위 제한
            distances.append(noisy_dist)
        distances = [sum(distances[i*18:18*(i+1)])/18  for i in range(self.num_beam)] # 180개를 10개 데이터로 평균화(18개의)
        distances = [x / max_dist for x in distances] # 정규화 과정!
        # print(f"ang : {distances}, len : {len(distances)}") # 이 180개의 거리 값을 10개로 압축/ 1도간격으로 18개의 평균을 내는거지...
        # print()
        return angles, distances



    def generate_maze_obstacles(self,agent_start,
                            num_walls=20,
                            wall_length_range=(5, 10),
                            wall_thickness=0.15, # 수정 0.04에서 0.15로 크게 영향은 없을듯?
                            min_clearance=0.5,
                            surround_size=3.5):
    # """
    # 랜덤한 미로형 장애물 (벽) 생성
    #   - num_walls: 생성할 벽 개수
    #   - wall_length_range: 벽 길이 범위 (m)
    #   - wall_thickness: 벽 두께 (m)
    #   - min_clearance: 벽 간 최소 거리 (m) — 이 값 이내로 가까워지면 겹친 것으로 간주하고 버림
    # """
        obstacles = []
        attempts = 0
        max_attempts = num_walls * 20
        
        # goal 좌표
        gx, gy = self.radiation.S_x, self.radiation.S_y
        ax, ay = agent_start

       
        # source 기준으로 원형으로 둘러싼 장애물..

        # circle_radius = surround_size / 2
        # gap_angle = self.np_random.uniform(0, 2 * math.pi)
        # gap_width = math.pi / 3
        # num_segments = 24
        # angle_step = 2 * math.pi / num_segments

        # for i in range(num_segments):
        #     start_angle = i * angle_step
        #     end_angle = (i + 1) * angle_step
        #     mid_angle = (start_angle + end_angle) / 2

        #     wrapped_diff = (mid_angle - gap_angle + math.pi) % (2 * math.pi) - math.pi
        #     if abs(wrapped_diff) < gap_width / 2:
        #         continue

        #     p1 = (gx + circle_radius * math.cos(start_angle), gy + circle_radius * math.sin(start_angle))
        #     p2 = (gx + circle_radius * math.cos(end_angle), gy + circle_radius * math.sin(end_angle))
        #     obstacles.append((p1, p2, wall_thickness))
        # 근원지를 완전히 둘러싸는 원형 장애물 생성 (구멍 없음)

        circle_radius = surround_size / 2
        num_segments = 24  # 원 둘레를 구성할 세그먼트 수 (15도 간격)
        #angle_step = 2 * math.pi / num_segments
        circle_points = [
            (gx + circle_radius * math.cos(angle),
             gy + circle_radius * math.sin(angle))
            for angle in np.linspace(0, 2 * math.pi, num_segments, endpoint=False)
        ]
        for i in range(num_segments):
            # angle1 = i * angle_step
            # angle2 = (i + 1) * angle_step
            # p1 = (gx + circle_radius * math.cos(angle1),
            #       gy + circle_radius * math.sin(angle1))
            # p2 = (gx + circle_radius * math.cos(angle2),
            #       gy + circle_radius * math.sin(angle2))
            p1 = circle_points[i]
            p2 = circle_points[(i + 1) % num_segments]
            obstacles.append((p1, p2, wall_thickness))
            obstacles.append((p1, p2, wall_thickness))


        return obstacles
        
    ###=========================================== Graphical result rendering ===================================================            
    def render_background(self, mode='human'): # 가스 확산 모델을 시각화 하는 함수...
        # print(8888888888888888888888888888) # 출력이 안됨.. 수정이 안되나? 이건 교수님께 질문.... /같은 모듈인데 경로가 달라서 그런거임!
        size = self.screen_height / 500
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.background_viewer is None:
                self.background_viewer = rendering.Viewer(self.screen_width, self.screen_height) # 그래픽창 생성!
            max_conc = 100; # 측정 농도값의 최대값!

            for xx in range(self.court_lx):
                for yy in range(self.court_ly): # 이게 그러면 x방향으로 한번쭉 도달 그다음 y올리고 다시 x도달시키고.. 반복
                    conc = isotropic_plume(xx+0.5, yy+0.5, self.gas, self.t_wind)
                    conc = self.np_random.normal(conc, math.sqrt(pow(self.env_sig,2) + pow(conc*self.sensor_sig_m,2)) )
                    # 이건 또 논문에서 정의한 대로 측정값의 분포를 정의함!
                    while conc < 0: #아 이건 While이라서 0이상이 될 때까지 반복하는 거임!
                        conc = self.np_random.normal(conc, math.sqrt(pow(self.env_sig,2) + pow(conc*self.sensor_sig_m,2)) )

                    x = xx*self.scale
                    y = yy*self.scale
                    plume = rendering.make_circle(4.5*size) # 아마 근원지를 원으로 표시하는 거겠지....
                    plume.add_attr(rendering.Transform(translation=(x, y)))  #plume을 (x,y)위치에 이동시키는 코드임!

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
        
    
    
    # def render_background_sample(self, mode='human'): # self.background_viewer를 그대로 쓰니깐 화면이 교대로 반복됨 그래서 새로운 객체를 정의함!
    #                                                   # self.background_viewer_radi로 새로 화면을 추가!
    #     size = self.screen_height / 500
    #     if mode == 'human':
    #         from gym.envs.classic_control import rendering
    #         if self.background_viewer_radi is None:
    #             self.background_viewer_radi = rendering.Viewer(self.screen_width, self.screen_height)  # 그래픽 창 생성
    #         max_dose = 50  # 시각화를 위한 최대 방사능 dose rate (예: 100 단위)  /100

    #         for xx in range(self.court_lx):
    #             for yy in range(self.court_ly):
    #                 # 기존의 isotropic_plume() 대신 방사능 dose rate를 계산하는 radiation_field() 함수를 사용
    #                 dose = radiation_field_for_render(xx + 0.5, yy + 0.5, self.radiation)
    #                 # 측정 오차를 반영하기 위해 노이즈 추가 (기존과 유사한 방식)
    #                 dose = self.np_random.normal(dose, math.sqrt(pow(self.env_sig, 2) + pow(dose * self.sensor_sig_m, 2)))
    #                 while dose < 0:
    #                     dose = self.np_random.normal(dose, math.sqrt(pow(self.env_sig, 2) + pow(dose * self.sensor_sig_m, 2)))

    #                 x = xx * self.scale
    #                 y = yy * self.scale
    #                 plume = rendering.make_circle(4.5 * size)
    #                 plume.add_attr(rendering.Transform(translation=(x, y)))

    #                 if dose > max_dose:  # 시각화를 위해 최대값을 제한
    #                     dose = max_dose
    #                     color = cm.jet(255)  # 최대값에 대응하는 색상
    #                     plume.set_color(color[0], color[1], color[2])
    #                     self.background_viewer_radi.add_onetime(plume)
    #                 elif dose > self.dose_eps + 0.5:
    #                     color_cal = round((math.exp(math.log(dose + 1) / math.log(max_dose + 1)) - 1) * 255)
    #                     if color_cal < 0:
    #                         color_cal = 0
    #                     color = cm.jet(color_cal)
    #                     plume.set_color(color[0], color[1], color[2])
    #                     self.background_viewer_radi.add_onetime(plume)

    #         return self.background_viewer_radi.render(return_rgb_array=(mode == 'rgb_array'))


    def render_background_sample(self, mode='human'):
        size = self.screen_height / 500
        if mode != 'human':
            return

        from gym.envs.classic_control import rendering
        if self.background_viewer_radi is None:
            self.background_viewer_radi = rendering.Viewer(self.screen_width, self.screen_height)

        max_dose = self.dose_max
        vis_frac     = 0.5
        radius_scale = 2.8 # 2.5
        step         = 0.5   # 1.0

        ld_min = math.log10(self.dose_eps + 1)
        ld_max = math.log10(max_dose * vis_frac + 1)

        # maze_obstacles = self.generate_maze_obstacles(agent_start=(self.radiation.S_x,self.radiation.S_y))
        xs = np.arange(0, self.court_lx, step)
        ys = np.arange(0, self.court_ly, step)

        for xx in xs:
            for yy in ys:
        # for xx in range(0, self.court_lx, step):
        #     for yy in range(0, self.court_ly, step):

                dose = radiation_field(xx + 0.5, yy + 0.5, self.radiation, self.obstacles)
                sigma = math.sqrt(self.env_sig**2 + (dose * self.sensor_sig_m)**2)
                noisy = self.np_random.normal(dose, sigma)
                dose = noisy if noisy >= 0 else dose

                ld   = math.log10(dose + 1)
                norm = (ld - ld_min) / (ld_max - ld_min)
                norm = max(0, min(1.0, norm))
                color_idx = int(norm * 255)

                x = xx * self.scale
                y = yy * self.scale
                plume = rendering.make_circle(radius_scale * size)
                plume.add_attr(rendering.Transform(translation=(x, y)))
                plume.set_color(*cm.jet(color_idx)[:3])
                self.background_viewer_radi.add_onetime(plume)

        for obs in self.obstacles:
            if len(obs) == 3:
                (x1, y1), (x2, y2), thickness = obs
            else:
                (x1, y1), (x2, y2) = obs

            length = math.hypot(x2 - x1, y2 - y1)
            num = max(int(length * 2), 1)
            for k in range(num + 1):
                frac = k / num
                ox = x1 + (x2 - x1) * frac
                oy = y1 + (y2 - y1) * frac
                dot = rendering.make_circle(3.0 * size)
                dot.set_color(0, 0, 0)
                dot.add_attr(rendering.Transform(
                    translation=(ox * self.scale, oy * self.scale)
                ))
                self.background_viewer_radi.add_onetime(dot)

        return self.background_viewer_radi.render(return_rgb_array='rgb_array')

    
    def render(self, mode='human'): # 
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
                for i in range(len(self.uav[n].dose_seq)):
                    measure = rendering.make_circle(math.pow(self.uav[n].dose_seq[i],1/10)*1.8*size) # 에이전트 원크기 / does_rate크기 따라서 커지고 작아짐! 원래 1/3 제곱임!
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


            goal = rendering.make_circle(7*size)
            goal.add_attr(rendering.Transform(translation=(self.radiation.S_x*self.scale,
                                                           self.radiation.S_y*self.scale)))
            goal.set_color(0.4, 0.6, 0.9)
            self.viewer.add_onetime(goal)

            #미로 장애물 렌더링 (generate_maze_obstacles 사용)
            # maze_obstacles = self.generate_maze_obstacles()
            for obs in self.obstacles:
                if len(obs) == 3:
                    (x1, y1), (x2, y2), thickness = obs
                else:
                    (x1, y1), (x2, y2) = obs

                length = math.hypot(x2 - x1, y2 - y1)
                num = max(int(length * 2), 1)
                for k in range(num + 1):
                    frac = k / num
                    ox = x1 + (x2 - x1) * frac
                    oy = y1 + (y2 - y1) * frac
                    dot = rendering.make_circle(3.0 * size)
                    dot.set_color(0, 0, 0)
                    dot.add_attr(rendering.Transform(
                        translation=(ox * self.scale, oy * self.scale)
                    ))
                    self.viewer.add_onetime(dot)
            
                    
        
            # x = self.uav[0].x
            # y = self.uav[0].y
            # heading = self.action_for_cast[0] * math.pi

            # angles, distances = self.simulate_lidar(x, y, heading) # 여기서 하드코딩 
            
            # print(f"dist : {distances}")
            # print()
            # if min(distances) < 0.5:
            #     print("crack!!!")
            #     print()

            # for theta, dist in zip(angles, distances):
            #     x2 = x + math.cos(theta) * dist
            #     y2 = y + math.sin(theta) * dist

            #     ray = rendering.Line((x * self.scale, y * self.scale), (x2 * self.scale, y2 * self.scale))
            #     ray.set_color(1.0, 0.6, 0.3)  # 주황색 LiDAR 빔
            #     ray.linewidth = 3
            #     self.viewer.add_onetime(ray)


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



            

