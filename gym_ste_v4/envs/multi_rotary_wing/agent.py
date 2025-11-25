import math
import numpy as np
from gym_ste_v4.envs.common.particle_filter import ParticleFilter # 3/19 시작... ok 완료..(action 제어는 제외, kmeans도 gmm이랑 원리는 같아서 제외함.)
from gym_ste_v4.envs.common.utils import segment_intersection
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')


class Agent:
    def __init__(self, env, init_pose, pf_num, mean_on, gmm_num, lidar_on ,kmeans_num, obs_normalization, num_disc_actions):
        ### Initial state that declared 
        self.x = init_pose[0] # init_pose 는 에이전트의 초킥 위치벡터 인듯...
        self.y = init_pose[1]
        self.pf_num = pf_num     # 논문에 파티클 정보와 가중치를 합산한 값을 파티클 필터의 평균이라 봄.. MVG데이터셋에 포함되는지 여부...
        self.mean_on = mean_on #  관측(observation) 벡터에 파티클 필터의 요약 통계(예: 파티클들의 평균 위치, 평균 방출 강도, 그리고 분산 값 등)를 포함할지 여부를 결정하는 불리언 플래그
        self.gmm_num = gmm_num # gmm 클러스터링 개수 
        self.kmeans_num = kmeans_num
        self.normalization = obs_normalization
        self.num_disc_actions = num_disc_actions # 이산행동 개수...?
        self.lidar_on = lidar_on
        self.lidar_val = [0] * env.num_beam # base의 observation space정의와 실제로 정의되는 agent사이에 차원이 맞아야 신경망 에러없음!!

        ## True state that agent should know / 결국 baseenv의 상태값을 수정하는 수 밖에...
        # self.gas = env.gas  # 이걸 이제  방사선 클래스로 수정!

        self.radiation = env.radiation # 방사선 객체!

        self.env_sig = env.env_sig # 바람에 의한 노이즈 >> 공기산란에 의한 노이즈/ 노이즈는 그냥 고정시켜놓고 쓰자! / 그리고 분포도 그냥 가우시안 쓰자!
        self.sensor_sig_m = env.sensor_sig_m #센서에 의한 노이즈. >> 노이즈 분산을 고정 시킬까? / 너무 깊게 들어가진 말자!
        self.court_lx = env.court_lx # 영역 제한
        self.court_ly = env.court_ly
        self.env = env

        # self.max_q = env.max_q  # 방출강도인데 이건... 방사선의 b=Ax 에서 x에 대응된다. 어차피 A구하는 공식있어서 b는 파티클에서 측정하면 되니깐...
        # self.max_radi_x = env.max_radi_x / 필요없음! 이미 활동도는 고정됨!

        self.step_size = env.step_size # 한번에 움직이는 거리
        self.max_step = env.max_step # 최대 스텝수(에피소드 당)
        self.motion_noise_std = env.motion_noise_std

        # self.conc_max = env.conc_max # 가스 농도 최댓값.  >> 선량률 최대값 b 로 수정!
        self.dose_max = env.dose_max   # 새롭게 방사선 선량률 공식으로 정의함(r=0.1)

        self.seed_num = env.seed_num 
        self.adjusted_action_on = env.adjusted_action_on

        self.obs_low_state = env.obs_low_state # obs의 하한과 상한... # 이건 이제 수정할것 최종확정한 뒤에 나중에 바꾸는 것으로...
        self.obs_high_state = env.obs_high_state
        ### Initial state that should be reset for new episode
        self.dur_t = 0      # duration time of out of plume / 플룸(가스 농도 최고 영역)에서 벗어난 지속 시간 / 어차피 리원드 줄때 쓰는거니깐 그대로 사용.

        self.last_dose_rate = 0 # 이전의 농도 측정값. >> b 로 수정  (11)
        self.highest_does_rate = 0 # 가장 높은 측정값.

        self.step_rew = 0 # 보상값.
        self.last_action = 0 # 이전의 행동. / 어차피 알고리즘은 그대로 가져갈 것이기 때문에, 그대로 두자...
        self.positions = [] 
        self.dose_seq = []  # 이런것도 그냥 이름만 바꾸고 그대로 가져가면 됨! (12)
        self.x_seq = [] 
        self.y_seq = [] 

        self.last_x = 0
        self.last_y = 0
        
        ### Functions
        self.np_random = env.np_random  # 여기서 env : BaseEnv 
        self.estimator = ParticleFilter(self) # 여기서 self : agent객체 /  파티클 필터의  args ; self(agent)
        self.pf_x = self.estimator.pf_x # 이 값들이 아마 소스의 위치와 분출강도?
        self.pf_y = self.estimator.pf_y
        self.pf_mass = self.estimator.pf_mass # 소스의 데이터(위치, 방출강도) >> 방사선 모델의 x 로 사용 / 솔직히 이것도 그냥 그대로 가져다 쓰는데 단위가 다르니 그것만 나중에 다시 생각!
        # (13)
        self.Wpnorms = self.estimator.Wpnorms # 파티클의 가중치 (아마도 1/n으로 처음에는 균등분포로 초기화될듯?)
        self.max_mass = self.estimator.max_cs_137_mass
        

        seed_for_cluster = 1
        if self.gmm_num > 0: # 아 여기서 gmm객체 정의했네... / MVG 데이터셋도 그냥 가져다 쓸거니깐...ㅇㅇ
            self.gmm = GaussianMixture(n_components=self.gmm_num, n_init=1, max_iter=20, random_state=seed_for_cluster)
            self.gmm_mean_x = np.ones(self.gmm_num)
            self.gmm_mean_y = np.ones(self.gmm_num)
            self.gmm_cov_x = np.ones(self.gmm_num)
            self.gmm_cov_y = np.ones(self.gmm_num)
            self.gmm_weights = np.ones(self.gmm_num)
        if self.kmeans_num > 0:
            self.kmeans = KMeans(n_clusters=self.kmeans_num, n_init=1, max_iter=20, random_state=seed_for_cluster)


    def _normalize_observation(self, obs): # 아마 관측 상태데이터가 들어가지 않을까? 각 요소를 정규화 시켜주는 작업이 될듯하다...
        normalized_obs = []
        for i in range(0, obs.size):
            normalized_obs.append((obs[i]-self.obs_low_state[i])/(self.obs_high_state[i] - self.obs_low_state[i]))
        return np.array(normalized_obs) # 내 예상대로 요소별로 각각 정규화를 하는거 같다 / (x-min) / (max-min)


    def _wind_sensor(self, true_wind_phi, true_wind_speed): # uniform은 균등분포임. 샘플의 추출확률이 동일하다. >> 방사선 모델에서는 바람을 고려하지 않음!
        wind_degree_fluc = 5  #25 #degree / 내생각엔 fluc가 풍향과 풍속의 노이즈가 아닐까 싶다. >> 그럼 굳이 사용할 필요가 있을까? 어차피 환경 노이즈가 있다고 가정하고 퉁치자!
        wind_speed_fluc = 0.1 #1  어차피 바람정보 안쓸거라서 굳이 필요없을거 같음.!
        wind_dir = self.np_random.uniform(low=(true_wind_phi-wind_degree_fluc)/180,
                                         high=(true_wind_phi+wind_degree_fluc)/180)
        if wind_dir < 0: # 아 0~2로 잡느 이유가 어차피 pi곱하면 0~2*pi 일껀데 그러면 0~360도 니깐 !!
            wind_dir += 2 #마이너스각도면 360에 가깝게 (290도 이런식 / 원래는 -70겠지)
        elif wind_dir > 2:
            wind_dir -= 2 # ( 20도 / 원래는 380도 그냥 1바퀴를 없앤다!)
        # wind_dir [radian]
        wind_speed = self.np_random.uniform(low=true_wind_speed-wind_speed_fluc, 
                                           high=true_wind_speed+wind_speed_fluc)
        return wind_dir*math.pi, wind_speed #  리턴 전까지는 degreee/180이여서 그냥 도였지만 pi를 곱하면서 radian으로 변환 ( 1rad = pi/180 degree)

    # def _gas_measure(self, true_conc): # true_conc : 논문의 m(p|seta) / 이 값이 아마도 노이즈가 없는 상태에서의 평균 농도를 구한것임! 
    #     env_sig = self.env_sig #1.0 #0.2 #0.4   # 바람 노이즈의 표준편차. # 여기서 true_conc : 방사선 선량률 b 임!(장애물 없다고 가정/ 거리제곱만 고려!)
    #     sensor_sig_m = self.sensor_sig_m #0.5 #0.1 #0.2 # 센서의 표준편차.

    #     conc_env = self.np_random.normal(true_conc, env_sig) # 아마도 바람 노이즈와 순수 가스농도를 합친다. 그냥 감마선이 가우시안 분포를 따른다고 하고 이것을 그대로 차용하자!
    #     # 그러므로 true_conc를 구하는 함수만 utils 에 만들어놓거나 다른 메서드로 구현해놓으면댐!
    #     while conc_env < 0: conc_env =0
    #         # conc_env = self.np_random.normal(true_conc, env_sig)
    #     radiation_dose_rate = self.np_random.normal(conc_env, conc_env*sensor_sig_m) # 여기에 센서 노이즈까지 합쳐준다. / 논문에서는 단순한 바람,센서 노이즈분포를 평균농도와 더하는 것으로만 소개함!
    #     while radiation_dose_rate < 0: radiation_dose_rate =0
    #         #radiation_dose_rate = self.np_random.normal(conc_env, conc_env*sensor_sig_m)

    #     return radiation_dose_rate # 최종적인 가스 농도의 분포가 완성된다! / 노이즈가 포함된 센서의 농도 측정값!..
    



    def _radiation_measure(self, true_conc): # true_conc : 논문의 m(p|seta) / 이 값이 아마도 노이즈가 없는 상태에서의 평균 농도를 구한것임! 
        env_sig = self.env_sig #1.0 #0.2 #0.4   # 바람 노이즈의 표준편차. # 여기서 true_conc : 방사선 선량률 b 임!(장애물 없다고 가정/ 거리제곱만 고려!)
        sensor_sig_m = self.sensor_sig_m #0.5 #0.1 #0.2 # 센서의 표준편차.

        # conc_env = self.np_random.normal(true_conc, env_sig) # 아마도 바람 노이즈와 순수 가스농도를 합친다. 그냥 감마선이 가우시안 분포를 따른다고 하고 이것을 그대로 차용하자!
        # # 그러므로 true_conc를 구하는 함수만 utils 에 만들어놓거나 다른 메서드로 구현해놓으면댐!
        # while conc_env < 0: conc_env =0
        #     # conc_env = self.np_random.normal(true_conc, env_sig) #
        # radiation_dose_rate = self.np_random.normal(conc_env, conc_env*sensor_sig_m) # 여기에 센서 노이즈까지 합쳐준다. / 논문에서는 단순한 바람,센서 노이즈분포를 평균농도와 더하는 것으로만 소개함!
        # while radiation_dose_rate < 0: radiation_dose_rate =0
        #     #radiation_dose_rate = self.np_random.normal(conc_env, conc_env*sensor_sig_m)

        conc_env = true_conc + self.np_random.normal(0, env_sig)
        conc_env = max(conc_env, 0)
        lam = conc_env * max(1 + sensor_sig_m, 1e-6)
        lam = max(lam, 1e-8)

        return self.np_random.poisson(lam)  # 최종적인 가스 농도의 분포가 완성된다! / 노이즈가 포함된 센서의 농도 측정값!..
    
    



    def _estimator_update(self, radiation_dose_rate, x, y, num_connected): # 아마도 노이즈 센서 측정값을 기준으로 우도함수 적용 후 가중치 만들고 
        self.estimator._weight_update(radiation_dose_rate, x, y,                           # 필요에 따라 리샘플링을 진행하는 작업을 거칠것(분포가 한쪽으로만 쏠리면..)
                                      self.pf_x, self.pf_y,self.pf_mass, self.Wpnorms, num_connected) # 여기서 self.estimator는 filter 이다
        self.pf_x = self.estimator.pf_x # 이때의 pf_x,y,q 값은 리샘플링을 거친 분포일것.. / 순차적 리샘플링 사용!
        self.pf_y = self.estimator.pf_y
        self.pf_mass = self.estimator.pf_mass
        self.Wpnorms = self.estimator.Wpnorms # 가중치도 정규화됨! / 그럼 여기서 바람정보만 빼고 conc >> dose rate로 수정
        # 즉 파티클 필터에서 가중치 계산하는 부분에서 바람을 고려안하고 변수만 바꿔주면 된다는 소리이다.
    def _observation(self, radiation_dose_rate): #역시 여기서도 바람을 고려하지 않는다.
        #moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
        
        if radiation_dose_rate > self.highest_does_rate: # 현재 얻은 측정값이 기존의 측정된 최댓값보다 크면 바로 수정한다.
            self.highest_does_rate = radiation_dose_rate # dose rate(방사선 선량률)로 수정!
            self.dur_t = 0 #$ self.dur_t가 10이라는 것은  step을 10번반복했지만 아직도 이전보다 더 높은 측정값을 수집하지 못했다는 것임!
            self.renew = True # 아마 리워드 일것
        else:
            self.dur_t += 1  # 그래서 측정 농도를 업데이트하지 못하면, 시간이 1증가한다.
            self.renew = False
        
        self.pf_x = self.estimator.pf_x
        self.pf_y = self.estimator.pf_y
        self.pf_mass = self.estimator.pf_mass  # does rate로 수정!
        self.Wpnorms = self.estimator.Wpnorms # 그냥 소스값과 파티클 가중치 정의. 


        self.mean_x = sum(self.pf_x * self.Wpnorms) # 이게 그거다. 논문에서 나온 gmm클러스터링 데이터셋의 m_t에 해당하는 것!(파티클과 가중치곱의 합산!)
        self.mean_y = sum(self.pf_y * self.Wpnorms)
        self.mean_mass = sum(self.pf_mass * self.Wpnorms) # does rate로 수정!
        self.CovXxp = np.var(self.pf_x) # 이것이 그 각 파티클 분포의 공분산의 대각성분의 곱인줄 알았으나(논문에서는 벡터형태로 써놓긴함)
        self.CovXyp = np.var(self.pf_y) # 하지만 친절하게도 각성분을 모두 고려했기 때문에, 각각은 스칼라값, 즉 각 성분의 평균과 분산이 되는것임!
        self.CovX_mass = np.var(self.pf_mass) # 대신 pf_x자체는 스칼라지만 여러성분을 모으면 벡터가 되겠지?
        
        self.pf_center = [self.mean_x, self.mean_y] # 그럼 각 파티클 분포의 중심은 당연히 평균 벡터가 되겠지...

        obs = np.array([                               #  [ut , ψt , ct−1 , ct , ch , tt , at−1 , pt ]을 구현한것 (논문)
                        float(self.dur_t),            # 단지 관측 데이터에서 바람은 뺴고 does rate 값로 교체하면 됨!  
                        float(self.x), float(self.y), # radiation_dose_rate >> radi_does_rate
                        float(self.last_action),
                        float(self.last_dose_rate), float(radiation_dose_rate), float(self.highest_does_rate)]) # 관측 데이터에서 바람 제외!

        if self.mean_on:  # 아까 예상한 대로 이건 obs에 gmm클러스트링의 m_t , 분산을 추가하는 불리언 이다./ 엄밀히 따지면 파티클과 가중치의 합산이긴 하지만,
            obs_mean =  [float(self.mean_x), float(self.mean_y), float(self.mean_mass),
                         float(self.CovXxp), float(self.CovXyp),float(self.CovX_mass)]

            obs=np.append(obs, obs_mean) # 넘파이 어레이에 요소를 추가하는 작업..


        if self.lidar_on:
            obs=np.append(obs, self.lidar_val)

        if self.gmm_num > 0: 
            if self.estimator.update_count == 0: # only update after resampling / 리샘플링을 수행한 경우에만 사용가능!
                pf_does_rate = np.column_stack((self.pf_x, self.pf_y)) # 예를 들어, pf_x = [10, 12, 30, 32]와 pf_y = [20, 22, 40, 42]라면 각 행벡터를 열벡터로 만들어서 axis=1방향으로 concat함!
                gmm_labels = self.gmm.fit_predict(pf_does_rate) # GMM 모델을 pf_does_rate 데이터에 맞춰 학습하고, 각 파티클이 어느 클러스터에 속하는지 라벨을 할당합니다
                self.gmm_weights = self.gmm.weights_ # 논문에서 봤던 가우시안 분포들의 혼합비율/ pi가 여기에 해당하지 않을까?..
            # gmm_label : np.array([0, 0, 1, 1])이런식으로 각 파티클의 (x,y)데이터를 기준으로 이것들이 어디에 속하는지 정해줌 (클러스터2개라서 0,1로...)
                self.gmm_data = []
                for k in range(self.gmm_num): # 결국 파티클 집단을 특정 개수의 클러스터링 집단으로 압축! (평균/ 분산구함)
                    self.gmm_data.append(pf_does_rate[gmm_labels == k]) # 어차피 pf_con은 x,y데이터를 열벡터로 붙힌 데이터셋임. gmm==0이면 . [T,T,F,F]가 되고 이건 각 샘플을 뽑아낼수 있는 마스크가 된다.
                    gmm_Wpnorms = self.Wpnorms[gmm_labels == k] # 행렬이 아닌 일반벡터도 마스킹이 가능한가? ㅇㅇ  1차원 벡터도 마스킹 가능함.
                    gmm_Wpnorms = gmm_Wpnorms/sum(gmm_Wpnorms) # 정규화 / 하지만 마스킹시 주의할 점은 불리언 값들이 샘플의 개수와 일치해야 한다는 점이다!
                    data_split = np.transpose(self.gmm_data[k]) # 이때 gmm_data는 [[k lablel에 해당하는 샘플 데이터셋] [k+1].... ]
                                                                # 그래서 self.gmm_data[k]는 리스트에서 그냥 순차적으로 데이터셋 불러오는거네.
                    self.gmm_mean_x[k] = sum(data_split[0] * gmm_Wpnorms) #data_split[0] : 행벡터(샘플)다들고 오는데 그게 x성분전체임!
                    self.gmm_mean_y[k] = sum(data_split[1] * gmm_Wpnorms) # data_split[1] : y성분 전체임..
                    # 가중치와 파티클 좌표를 곱하고 합산한것이 기댓값이 되는 이유는 가중치를 파티클의 분포로 해석하는 것이다.! 그러면 납득이 됨!
                    
                    #self.gmm_mean_x = np.ones(self.gmm_num) 에서 알 수 있듯이, 미리 공간을 만들어 두었고 순차적으로 인덱스를 늘려가며 공간을 채울수 있다(0,1,2....)
                    if np.shape(data_split[0])[0] == 0: # 여기서 data_split의 차원은 (2,num of resampled particle) / 2:x,y  
                        self.gmm_cov_x[k] = 0 # 첫번째 차원이 0이라는 소리는 데이터가 없다는 말과 같은것이 아닐까? 그 클러스터에는?
                        self.gmm_cov_y[k] = 0
                    else:
                        self.gmm_cov_x[k] = np.var(data_split[0]) # 각 클러스터의 파티클에 대한 x,y의 분산계산
                        self.gmm_cov_y[k] = np.var(data_split[1]) # 이건 여러 개의 파티클 집단을 하나의 분산과 평균으로 나타낼 수 있다. 
                                                                  
            for k in range(self.gmm_num): # MVG: st = [mt , diag(Σt ), M t , Σt,diag , Πt ]의 구현... >> 이건 그대로 가져가기!
                obs = np.append(obs, [float(self.gmm_mean_x[k]), float(self.gmm_mean_y[k]), # 아마 각 요소들은 스칼라 일것..
                                      float(self.gmm_cov_x[k]), float(self.gmm_cov_y[k]),
                                      float(self.gmm_weights[k])]) # 이것이 바로 gmm클러스터링 데이터(gmm평균, 분산, 가중치...)

        if self.kmeans_num > 0: # k-means 사용여부 / 이건 gmm vs k-means 선택문제 같음(코드흐름이 정확히 일치함 단지 tool을 뭘 사용할것이냐의 차이!)
            if self.estimator.update_count == 0: # only update after resampling
                pf_does_rate = np.column_stack((self.pf_x, self.pf_y))
                kmeans_labels = self.kmeans.fit_predict(pf_does_rate)

                self.km_mean_x = np.ones(self.kmeans_num)
                self.km_mean_y = np.ones(self.kmeans_num)
                self.km_cov_x = np.ones(self.kmeans_num)
                self.km_cov_y = np.ones(self.kmeans_num)
                self.km_data = []
                for k in range(self.kmeans_num):
                    self.km_data.append(pf_does_rate[kmeans_labels == k])
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
                obs = np.append(obs, [float(self.km_mean_x[k]), float(self.km_mean_y[k]),
                                      float(self.km_cov_x[k]), float(self.km_cov_y[k])] )


        self.last_dose_rate = radiation_dose_rate
        
        self.dose_seq.append(radiation_dose_rate)
        self.x_seq.append(self.x)
        self.y_seq.append(self.y)
        self.positions.append([self.x, self.y])
        self.cov_val = np.sqrt(self.CovXxp/pow(self.court_lx,2) + \
                               self.CovXyp/pow(self.court_ly,2) +\
                                self.CovX_mass/pow(self.max_mass,2)   ) # 이게 정확히 뭐지? 활동도가 너무크나?

        if self.normalization:
            obs = self._normalize_observation(obs)

        return obs
        
    def adjusted_action(self, action, neighbor_pose): # 이것들은 멀티 에이전트에 필요한 내용이므로 굳이 다루지는 않겠다.
        avoidance_r = self.step_size*1.05
        #print("Neigbor Position: ", neighbor_pose)
        nei_pose = np.array(neighbor_pose)
        forbidden_range = []
        action_adj = action
        for pose in nei_pose: # Add all forbidden region because of other agents
            vec = pose - [self.x, self.y]
            dist = np.linalg.norm(vec)
            #print("Distance before move: ", dist)
            if dist < avoidance_r*2:
                dir = math.atan2(vec[1], vec[0])
                dir_alpha = math.acos( ((dist-self.step_size)/2)/avoidance_r )
                new_range = [dir-dir_alpha, dir+dir_alpha]
                forbidden_range.append(new_range)
                for i in range(len(forbidden_range)-1):
                    if (forbidden_range[i][0] <= new_range[1] and new_range[0] <= forbidden_range[i][1]):
                        forbidden_range[i][0] = np.min([new_range[0],forbidden_range[i][0]])
                        forbidden_range[i][1] = np.max([new_range[1],forbidden_range[i][1]])
                        forbidden_range.pop()

        for i in range(len(forbidden_range)): # If the region exceeded -pi to +pi
            range_exceeded = False
            if forbidden_range[i][0] < -math.pi:
                new_low  = forbidden_range[i][0] + 2*math.pi
                new_high = forbidden_range[i][1] + 2*math.pi
                range_exceeded = True
            elif forbidden_range[i][1] > math.pi:
                new_low  = forbidden_range[i][0] - 2*math.pi
                new_high = forbidden_range[i][1] - 2*math.pi
                range_exceeded = True
            if range_exceeded == True:
                new_range = [new_low, new_high]
                if forbidden_range[i][0] <= new_range[1] and new_range[0] <= forbidden_range[i][1]:
                    forbidden_range[i][0] = np.min([new_range[0],forbidden_range[i][0]])
                    forbidden_range[i][1] = np.max([new_range[1],forbidden_range[i][1]])
                else:
                    forbidden_range.append(new_range)

        for single_range in forbidden_range:
            #print("SINGLE RANGE: ", single_range)
            if single_range[0]/math.pi < action and action < single_range[1]/math.pi:
                action_adj = single_range[0]/math.pi # Each agent move to right turn when encounter other agents

        #print("Action: ", action)
        #print("Forbidden Range: ", np.array(forbidden_range)/math.pi)
        #print("Adjusted Action: ", action_adj)
        return action_adj

    def _calculate_position(self, action, neighbor_pose):
        if self.adjusted_action_on:
            action_adj = self.adjusted_action(action, neighbor_pose)
        else:
            action_adj = action
        if self.num_disc_actions > 0:
            disc_action = np.argmax(action_adj)/self.num_disc_actions
            angle = (disc_action) * math.pi 
            self.last_action = disc_action
        else:
            angle = (action_adj) * math.pi
            #self.last_action = action_adj

        # calculate new agent state
        prev_x, prev_y = self.x, self.y
        target_x = self.x + math.cos(angle) * self.step_size
        target_y = self.y + math.sin(angle) * self.step_size

        noisy_x = target_x + self.np_random.normal(0, self.motion_noise_std)
        noisy_y = target_y + self.np_random.normal(0, self.motion_noise_std)

        noisy_x = min(max(noisy_x, 0), self.court_lx)
        noisy_y = min(max(noisy_y, 0), self.court_ly)

        closest_point = None
        closest_t = None
        for obs in getattr(self.env, "obstacles", []) or []:
            (ox1, oy1), (ox2, oy2), *_ = obs
            result = segment_intersection((prev_x, prev_y), (noisy_x, noisy_y), (ox1, oy1), (ox2, oy2))
            if result:
                point, t = result
                if closest_t is None or t < closest_t:
                    closest_t = t
                    closest_point = point

        if closest_point is not None:
            epsilon = 1e-3
            backoff_t = max(0.0, closest_t - epsilon)
            noisy_x = prev_x + (noisy_x - prev_x) * backoff_t
            noisy_y = prev_y + (noisy_y - prev_y) * backoff_t

        self.x = noisy_x
        self.y = noisy_y