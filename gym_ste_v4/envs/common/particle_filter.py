# import numpy as np
# import math
# from scipy.special import gammaln

# class ParticleFilter: # 그럼 이 코드를 통해서 정확히 파티클 필터가 어떻게 작동하는지 알 수 있겠군.... 

#     def __init__(self, args):  # 여기서  args : agent의 객체임.. 이것의 self를 필터에 대입함.. 그러면 init에 들어감!!
#         self.update_count = -1
#         self.sensor_sig_m = args.sensor_sig_m # 센서의 노이즈 >> 노이즈는 그대로 가져간다.. 어차피 고정해서 쓸거임!
# #        print("sig_m: ", self.sensor_sig_m)
#         self.env_sig = args.env_sig        # 바람에 의한 노이즈
# #        print("env_sig: ", self.env_sig)
#         self.pf_num = args.pf_num    # 파티클 개수 

#         self.measurement_interval = getattr(args, "delta_t", 1.0) # 이건 추후에 다시 결정..
#         self.counts_per_uSv_per_h = getattr(args, "counts_per_uSv_per_h", 3.0)

#         self.radiation = args.radiation  # 가스 특성? >> args.radiation 으로 수정할꺼임/ 아마도 근본적으로는 baseenv의 속성값을 수정해야 할것...

#         self.max_cs_137_mass = 9.47 * 10**(-6) # 10^7 nsv/h 를 거리 r=1m로 계산할때 필요한 질량
#         self.min_cs_137_mass = self.max_cs_137_mass / 2 # 이건 아마 같은거리일때 선량률이 절반인 질량이겠지.. 
#         self.special_activity = 3.2*10**12 # 반감기 (30.17)년에 따른 specific activity: Bq/g
#         self.cs_137_gamma = 330/(1000) # kBq에서 10^3만 나눠주자..
#         self.court_lx = args.court_lx   # 탐색영역?
#         self.court_ly = args.court_ly

# #        self.pf_x = np.ones(self.pf_num)*np.nan
# #        self.pf_y = np.ones(self.pf_num)*np.nan
# #        self.pf_source_activity = np.ones(self.pf_num)*np.nan
# #        self.Wpnorms = np.ones(self.pf_num)*np.nan

#         self.np_random = args.np_random 

#         pf_low_state_x = np.zeros(self.pf_num) # particle filter (x1,x2,x3, ...) 파티클 개수만큼 정의됨 / 각 파티클이 source의 후보니깐! 
#         pf_low_state_y = np.zeros(self.pf_num) # particle filter (y1,y2,y3, ...)
#         pf_low_mass = np.ones(self.pf_num) * self.min_cs_137_mass
#         pf_low_state_wp = np.zeros(self.pf_num) # particle filter (w1,w2,w3, ...)

#         pf_high_state_x = np.ones(self.pf_num)*self.court_lx 
#         pf_high_state_y = np.ones(self.pf_num)*self.court_ly
#         pf_high_mass = np.ones(self.pf_num)*self.max_cs_137_mass # 이제는 질량에 노이즈를 줘서 선량률 변화를 이끌어내보자!
        
#         pf_high_state_wp = np.ones(self.pf_num)  # 가중치만 0~1로 비율처럼 설정 >> 즉 파티클의 분포로 해석된다

#         self.pf_x = self.np_random.uniform(low = pf_low_state_x, high = pf_high_state_x) # 균등분포로 파티클의 위치와 방출강도를 뽑는다.
#         self.pf_y = self.np_random.uniform(low = pf_low_state_y, high = pf_high_state_y)
#         self.pf_mass = self.np_random.uniform(low = pf_low_mass, high = pf_high_mass) # 활동도도 분포로 가져가보자!
#         # self.pf_source_activity = self.np_random.uniform(low = pf_low_state_radi_x, high = pf_high_state_radi_x) # 그냥 q >> radi_x로 !
#         self.Wpnorms = np.ones(self.pf_num)/self.pf_num # 가중치 정규화하는 과정인듯? 어차피 각 파티클에 곱해져서 sum 하는 거임!


#     def _pf_does_rate(self, agent_x, agent_y, source_x, source_y, source_mass): # 아 실제 근원지가 아니라 pf의 위치구나!
#         avoid_zero = (np.sqrt(pow(source_x - agent_x,2) + pow(source_y - agent_y,2) ) < 1e-50) # bool
#         source_x[avoid_zero] += 1e-50 # source는 별게 아니라 pf 를 의미하는 것일것....
#         source_y[avoid_zero] += 1e-50
#         # dist = np.sqrt(pow((source_x - agent_x), 2) + pow(source_y - agent_y, 2)) # wind_d : 바람 방향각도
#         # y_n = -(agent_x - source_x)*math.sin(wind_d)+ \
#         #        (agent_y - source_y)*math.cos(wind_d)
#         # lambda_plume = math.sqrt(self.radiation.d * self.radiation.t / (1 + pow(wind_s,2) * self.radiation.t/4/self.radiation.d) ) # radiation.t : tau
#         # conc_com_1 = source_mass/(4 * math.pi * self.radiation.d * dist)     #             
#         # conc_com_2 = np.exp( -y_n * wind_s/(2*self.radiation.d) - dist/lambda_plume)
#         # conc = conc_com_1 * conc_com_2
#         dx = agent_x - source_x  # raidation.S_x만 env에 따로 정의해놓으면댐 가스께 복붙해서...
#         dy = agent_y - source_y
#         r = np.sqrt(dx*dx + dy*dy) # 넘파이는 np.sqrt로 계산 math는 스칼라만 가능!
    
#         # To avoid division by zero in case the detection point is extremely close to the source:
#         # if r < 1e-6:  #넘파이는 if못씀 .. 여러개 있을 때 true,false일수도 있어서 대신 마스킹을 사용함!
#         #     r = 1e-6
#         r = np.maximum(r, 1e-6)  # 넘파이에서 안전하게 최소값을 확보하는 방법은 클리핑을 하는것이다. 1e-6 이하면 1e-6으로 아니면 자기값 가짐!
    
#         dose_rate = self.cs_137_gamma * self.special_activity *  source_mass / (r ** 2) # kBq 단위
                                           
#         return dose_rate
    
#     def _dose_rate_to_cps(self, dose_rate):
#         dose_rate = np.maximum(dose_rate, 0.0)
#         dose_rate_uSv_per_h = dose_rate / 1000.0
#         return dose_rate_uSv_per_h * self.counts_per_uSv_per_h

    
#     def _sensor_rate(self, expected_rate):
#         expected_counts = self._dose_rate_to_cps(expected_rate) * self.measurement_interval
#         env_counts = self._dose_rate_to_cps(self.env_sig) * self.measurement_interval
#         scaled_counts = expected_counts * np.maximum(1.0 + self.sensor_sig_m, 1e-6)
#         return np.maximum(scaled_counts + env_counts, 1e-8)

#     def _poisson_likelihood(self, observation, rate):
#         safe_rate = np.maximum(rate, 1e-8)
#         log_likelihood = observation * np.log(safe_rate) - safe_rate - gammaln(observation + 1)
#         log_likelihood = np.nan_to_num(log_likelihood, nan=-200, posinf=-200, neginf=-200)
#         likelihood = np.exp(log_likelihood)
#         likelihood[likelihood < 1e-200] = 1e-200
#         return likelihood


#         # 이게 그 각 파티클에 대한 예측 농도값? 인듯.. >> 이거 그냥 방사선 선량률 공식 사용해서 구하면 됨... 거리 제곱에 반비례만 이용!
#         # 어차피 이건 파티클에 대한 예측 측정값이라고 생각하면 된다.. 어차피 이거 평균으로 실제 측정값을 구할수 있다. 그리고 우도함수에서도 사용됨(가우시안)

#     def _weight_calculate(self, radiation_measure, agent_x, agent_y, pf_x, pf_y, pf_mass):
#         self.radiation_measure = radiation_measure # 이건 센서 측정 농도값.
#         self.agent_x = agent_x
#         self.agent_y = agent_y
#         # self.wind_d = wind_d # 풍향  / 바람정보 고려안함! 그냥 주석처리만 해놓을꺼임/ 어차피 새로만들꺼라 상관은 없긴해..
#         # self.wind_s = wind_s # 풍속

#         pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass) # 파티클에 대한 예측 농도. >> 예측 방사선 선량률..


        

#         lambda_rate = self._sensor_rate(pf_dose_rate)
#         poisson_weight = self._poisson_likelihood(self.radiation_measure, lambda_rate)

#         poisson_weight[poisson_weight != poisson_weight] = 1e-200
#         poisson_weight[poisson_weight < 1e-200] = 1e-200
#         return poisson_weight  # 어차피 사실 변수명만 바꾸고 함수 몇개만 바꾸면 원래꺼랑 별차이는 안남 수치를 그대로 들고 오기때문에...


#     def _particle_resample(self, gauss_new):
#             N = self.pf_num # 파티클 개수
#             M = N  # 리샘플링후에도 개수는 동일해야함!
#             indx = np.ones(N)*-1 # 인덱스는 0인 n차원 벡터.
#             Q = np.cumsum(self.Wpnorms) # 가중치 누적합.
#             indx = np.zeros(N) # N=5라 한다면 np.arange(N)/N : [0,0.2,0.4,0.6,0.8]이됨(0.2로 일정함)
#             # 그리고 self.np_random.uniform(0,1/N, N)는 (0~0,2)사이에서 5개의 난수를 생성함/그리고 그것을 0.2간격이랑 더함!
#             # 결과는 [0.12,0.25,0.58,0.69,0.95]같은 thersold가 나올 것이고 일종의 경계값 역할을 한다.
#             T = np.arange(N)/N + self.np_random.uniform(0,1/N, N)
#             i=0
#             j=0
#             while(i<N and j<M):
#                 while(Q[j] < T[i]): # 이게 경계면은 하나로 고정하고 가중치 누적합을 계속 올리는..
#                     j = j+1 # 만약 누적합 (j=0) 이 첫번째 경게보다 크면 (i=0), j 업데이트 안하고 바로 리샘플링 가중치로 저장함!
#                 indx[i]=j  # j가 계속 T보다 크다면, j는 계속 저장할 수 있다! 
#                 i=i+1

#             indx = np.int64(indx) # 이 인덱스는 아마도 몇번째 가중치가 선택되었는지에 대한 것일듯 예를 들어 [0,1,3,3,3]이렇게 나올수 있지..
#                                   # 이 과정을 순차적 리샘플링 이라고 하는거 갇다.(본 거 같음!)
#             self.pf_x = self.pf_x[indx] # 그리고 가중치가 높은 파티클이 선택될 가능성이 높다 >> 중복선택될수 있다.
#             self.pf_y = self.pf_y[indx]
#             self.pf_mass = self.pf_mass[indx] # q >> x(방사선 방출강도) / 어차피 q만 잘 수정하면, 별 문제는 없을거 같다.

#             mm = 2
#             A=pow(4/(mm+2), 1/(mm+4) ) # 방사선 모델인 b=Ax에서의 A와는 당연히 다른값이다! 
#             cx = 4*math.pi/3
#             hopt = A*pow(A,-1/(mm+4)) # 그냥 상수 같은 것으로 간주.
#             for _ in range(1):
#                 CovXxp = np.var(self.pf_x) # 파티클 위치 및 방출강도의 분산! 
#                 CovXyp = np.var(self.pf_y)
#                 CovX_mass_xp = np.var(self.pf_mass)

#                 dkXxp = math.sqrt(CovXxp) # 
#                 dkXyp = math.sqrt(CovXyp)
#                 dkX_mass_xp = math.sqrt(CovX_mass_xp)
#                 # 이게 순차적 리샘플링을 거친 파티클 분포에 다시 노이즈를 준다.. 조금 더 새롭게 분포하게..
#                 nXxp = self.pf_x + (hopt*dkXxp*self.np_random.normal(0,1,self.pf_num) ) # hopt × dkXxp × noise 형식으로 파티클 샘플을 재정의(너무 붙어있는거 방지)
#                 nXxp[nXxp>self.court_lx] = self.court_lx # out of area
#                 nXxp[nXxp<0] = 0 # out of area

#                 nXyp = self.pf_y + (hopt*dkXyp*self.np_random.normal(0,1,self.pf_num) )
#                 nXyp[nXyp>self.court_ly] = self.court_ly # out of area
#                 nXyp[nXyp<0] = 0 # out of area
                

#                 nX_mass_xp = self.pf_mass + (hopt*dkX_mass_xp*self.np_random.normal(0,1,self.pf_num) ) # q >> x로 수정!
#                 nX_mass_xp[nX_mass_xp<0] = 0 # out of range

#                 # 이렇게 노이즈를 첨가한 파티클 샘플을 다시 가중치계산함. 이것들에 대한 우도함수를 재정의함!(리샘플링한 파티클은 또 다시 예측 분포로 봄!)
#                 n_new = self._weight_calculate(self.radiation_measure, self.agent_x, self.agent_y, nXxp, nXyp, nX_mass_xp)
#                 alpha = n_new/gauss_new[indx] # 새롭게 정의된 가중치와 이전의 중첩을 포함한 가중치간의 비율..
#                 mcrand = self.np_random.uniform(0,1,self.pf_num) # 파티클 개수만큼의 0~1의 난수.
# #                print(alpha > mcrand)
#                 new_point_bool = alpha > mcrand # 어차피 스무딩 가중치가 이전보다 좋다면 1이 넘을테니 thersold가 충분히 됨!
#                 self.pf_x[new_point_bool] = nXxp[new_point_bool] # 1보다 작다면 어차피 원래꺼랑 비슷해서 아무거나 써도 상관없음!
#                 self.pf_y[new_point_bool] = nXyp[new_point_bool]
#                 self.pf_mass[new_point_bool] = nX_mass_xp[new_point_bool] # 아 애초에 n_new로 pf를 업데이트 시키는구나...
#             self.Wpnorms = np.ones(self.pf_num)/self.pf_num # 아 그래서 다시 초기화 시키는 건가? 이미 리샘플링 할때 가중치 써먹었으니...
            

#     def _weight_update(self, measure, agent_x, agent_y, pf_x, pf_y , pf_mass, Wpnorms, num_connected): # 바람정보 고려하지않음!
#         #self.update_count += 1 # 근데 여기서도 어차피 weight계산을 함...
#         Wp_sum = 0
#         resample_true = False

#         self.agent_x = agent_x
#         self.agent_y = agent_y
#         self.radiation_measure = measure # 근원지에서 측정한 값을 써야되는거 아님?


#         # pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass) # 평균 가스 농도! >> 예측 방사선 선량률 
#         # mean_dose_rate = (pf_dose_rate + self.radiation_measure)/2 # 센서 노이즈 정의하는데 사용

#         pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass) # 평균 가스 농도! >> 예측 방사선 선량률
#         lambda_rate = self._sensor_rate(pf_dose_rate)
#         gauss_new = self._poisson_likelihood(self.radiation_measure, lambda_rate)
#         # mean_dose_rate = (pf_dose_rate + self.radiation_measure)/2 # 센서 노이즈 정의하는데 사용

        

#         gauss_new[gauss_new != gauss_new] = 1e-200
#         gauss_new[gauss_new < 1e-200] = 1e-200

#         sort_g = np.sort(gauss_new) # ort_g = np.sort(gauss_new)로 우도 값을 정렬하여, 모든 파티클의 우도 값이 동일한지 확인
#         if (sort_g[self.pf_num-1] == sort_g[0]): resample_true = True # 가장 큰 값과 작은 값이 동일하다면 이는 모든 값이 같으므로...리샘플링!
#         #if (self.update_count == 10): resample_true = True
#         Wps = Wpnorms * (gauss_new**(1/num_connected))
#         Wp_sum = np.sum(Wps)
     

#         self.pf_x = pf_x # 가중치 업데이트 시 정의된 입력 파티클의 위치 및 방출강도!
#         self.pf_y = pf_y # 근데 이건 초기의 입력값 아닌가? 다시 초기화 되는거 아님? 이것의 의도가 뭘까? / 교수님께 질문해보자...
       

#         Wpnorms = Wps/Wp_sum 


#         self.pf_x = pf_x # 가중치 업데이트 시 정의된 입력 파티클의 위치 및 방출강도!
#         self.pf_y = pf_y # 근데 이건 초기의 입력값 아닌가? 다시 초기화 되는거 아님? 이것의 의도가 뭘까? / 교수님께 질문해보자...
       
#         self.Wpnorms = Wpnorms

#         self.update_count += 1

#         if 1/sum(pow(Wpnorms,2)) < self.pf_num*0.5 or resample_true: # 1 for every time
#             self.update_count = 0 # 이건 리샘플링을 했다는 증거이다!
#             self._particle_resample(gauss_new) # self.pf_x, self.pf_y, self.pf_source_activity, self.Wpnorms이 업데이트...
#             #self.CovXxp = np.var(self.pf_x)
#             #self.CovXyp = np.var(self.pf_y)
#             #self.CovX_radi_xp = np.var(self.pf_source_activity)
#             #self.resampled = 1


#         return self.pf_x, self.pf_y, self.pf_mass, self.Wpnorms # 
    
#     # 3/18 : particle filter 내용분석 완료! 3/19 : Agent 내용 전체 분석목표! / 3/21 : 가스변수 >> 방사능 변수 변환완료 목표!

import numpy as np
import math
from scipy.special import gammaln

class ParticleFilter: 
    def __init__(self, args):
        self.update_count = -1
        self.sensor_sig_m = args.sensor_sig_m 
        self.env_sig = args.env_sig        
        self.pf_num = args.pf_num    

        # delta_t: 카운트 계산을 위한 시간 간격
        self.measurement_interval = getattr(args, "delta_t", 1.0) 
        self.counts_per_uSv_per_h = getattr(args, "counts_per_uSv_per_h", 3.0)

        self.radiation = args.radiation
        
        self.max_cs_137_mass = 9.47 * 10**(-6) 
        self.min_cs_137_mass = self.max_cs_137_mass / 2 
        self.special_activity = 3.2*10**12 
        self.cs_137_gamma = 330/(1000) 
        self.court_lx = args.court_lx   
        self.court_ly = args.court_ly

        self.np_random = args.np_random 

        # 파티클 초기화
        pf_low_state_x = np.zeros(self.pf_num) 
        pf_low_state_y = np.zeros(self.pf_num) 
        pf_low_mass = np.ones(self.pf_num) * self.min_cs_137_mass
        
        pf_high_state_x = np.ones(self.pf_num)*self.court_lx 
        pf_high_state_y = np.ones(self.pf_num)*self.court_ly
        pf_high_mass = np.ones(self.pf_num)*self.max_cs_137_mass 
        
        self.pf_x = self.np_random.uniform(low = pf_low_state_x, high = pf_high_state_x) 
        self.pf_y = self.np_random.uniform(low = pf_low_state_y, high = pf_high_state_y)
        self.pf_mass = self.np_random.uniform(low = pf_low_mass, high = pf_high_mass) 
        
        self.Wpnorms = np.ones(self.pf_num)/self.pf_num 

    def _pf_does_rate(self, agent_x, agent_y, source_x, source_y, source_mass): 
        # 거리 역제곱 법칙 (nSv/h 단위가 나오도록 물리상수 매칭 필요)
        dx = agent_x - source_x 
        dy = agent_y - source_y
        r = np.sqrt(dx*dx + dy*dy) 
        r = np.maximum(r, 0.01) # 0으로 나누기 방지 (최소 거리 1cm)
    
        dose_rate = self.cs_137_gamma * self.special_activity * source_mass / (r ** 2)                                   
        return dose_rate
    
    def _dose_rate_to_cps(self, dose_rate_nSv):
        # nSv/h -> uSv/h -> CPS 변환
        dose_rate_nSv = np.maximum(dose_rate_nSv, 0.0)
        dose_rate_uSv_per_h = dose_rate_nSv / 1000.0
        return dose_rate_uSv_per_h * self.counts_per_uSv_per_h

    def _get_lambda(self, expected_rate_nSv):
        # 파티클 예측 선량률 -> 기대 카운트(Lambda)
        # 1. 신호 성분
        signal_cps = self._dose_rate_to_cps(expected_rate_nSv)
        # 2. 배경 성분 (env_sig를 배경잡음 nSv/h로 가정)
        bg_cps = self._dose_rate_to_cps(self.env_sig) 
        
        # 총 기대 카운트 (Lambda)
        total_lambda = (signal_cps + bg_cps) * self.measurement_interval
        return np.maximum(total_lambda, 1e-9) # 로그 계산 오류 방지용 최소값

    def _poisson_likelihood(self, observation_nSv, expected_rate_nSv):
        # 1. 실제 관측값(nSv)을 정수형 카운트(k)로 변환
        obs_cps = self._dose_rate_to_cps(observation_nSv)
        k = np.round(obs_cps * self.measurement_interval) 
        
        # 2. 파티클 예측값(nSv)을 Lambda로 변환
        lam = self._get_lambda(expected_rate_nSv)

        # 3. Log-Likelihood 계산: k*ln(lam) - lam - ln(k!)
        log_likelihood = k * np.log(lam) - lam - gammaln(k + 1)
        
        # NaN 및 무한대 처리 (매우 중요!)
        log_likelihood = np.nan_to_num(log_likelihood, nan=-100.0, posinf=-100.0, neginf=-100.0)
        
        # exp 처리 시 Underflow 방지
        likelihood = np.exp(log_likelihood)
        likelihood = np.nan_to_num(likelihood, nan=1e-30) # NaN 방지
        
        return likelihood

    def _weight_calculate(self, radiation_measure, agent_x, agent_y, pf_x, pf_y, pf_mass):
        # 외부에서 호출 시 사용 (구조 유지)
        pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass)
        poisson_weight = self._poisson_likelihood(radiation_measure, pf_dose_rate)
        return poisson_weight

    def _particle_resample(self, likelihoods):
        # Systematic Resampling (기존 로직 유지하되 안전성 보강)
        N = self.pf_num 
        Q = np.cumsum(self.Wpnorms)
        indx = np.zeros(N, dtype=np.int64)
        T = np.arange(N)/N + self.np_random.uniform(0,1/N, N)
        
        i=0
        j=0
        while(i<N and j<N): # 인덱스 초과 방지
            while(j < N-1 and Q[j] < T[i]): 
                j = j+1 
            indx[i]=j 
            i=i+1
        
        self.pf_x = self.pf_x[indx]
        self.pf_y = self.pf_y[indx]
        self.pf_mass = self.pf_mass[indx]

        # Regularized Particle Filter (Jittering) - 파티클 다양성 확보
        # 분산이 너무 작으면 최소 노이즈 추가
        var_x = np.maximum(np.var(self.pf_x), 1e-4)
        var_y = np.maximum(np.var(self.pf_y), 1e-4)
        var_m = np.maximum(np.var(self.pf_mass), 1e-20)
        
        hopt = (4/(N+2))**(1/(2+4)) # Bandwidth (Simple assumption)
        
        # 위치 노이즈 추가
        self.pf_x += hopt * np.sqrt(var_x) * self.np_random.normal(0, 1, N)
        self.pf_y += hopt * np.sqrt(var_y) * self.np_random.normal(0, 1, N)
        self.pf_mass += hopt * np.sqrt(var_m) * self.np_random.normal(0, 1, N)

        # 맵 밖으로 나간 파티클 클리핑 (Clipping)
        self.pf_x = np.clip(self.pf_x, 0, self.court_lx)
        self.pf_y = np.clip(self.pf_y, 0, self.court_ly)
        self.pf_mass = np.maximum(self.pf_mass, 0) # 질량은 음수 불가

        # 리샘플링 후 가중치 초기화
        self.Wpnorms = np.ones(self.pf_num)/self.pf_num

    def _weight_update(self, measure, agent_x, agent_y, pf_x, pf_y , pf_mass, Wpnorms, num_connected):
        
        self.radiation_measure = measure 
        self.agent_x = agent_x
        self.agent_y = agent_y

        # 1. 우도 계산
        pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass)
        likelihoods = self._poisson_likelihood(self.radiation_measure, pf_dose_rate)

        # 2. 가중치 업데이트
        Wps = Wpnorms * likelihoods
        Wp_sum = np.sum(Wps)
        
        # [핵심 수정] NaN / 0 방어 코드
        # 모든 파티클의 확률이 0이 되어버린 경우 (Weight Collapse)
        if Wp_sum <= 1e-30 or np.isnan(Wp_sum):
            # 비상 대책: 가중치를 균등하게 리셋하고, 리샘플링을 강제함
            Wps = np.ones(self.pf_num) / self.pf_num
            Wp_sum = 1.0
            # (선택) 여기서 파티클을 맵 전체에 다시 뿌려주는 것이 좋을 수도 있음
            # 하지만 최소한 프로그램이 죽지는 않게 함
        
        # 정규화 (Normalize)
        Wpnorms = Wps / Wp_sum
        
        # 혹시라도 남았을 NaN 제거
        Wpnorms = np.nan_to_num(Wpnorms, nan=1.0/self.pf_num)
        
        # 상태 저장
        self.pf_x = pf_x
        self.pf_y = pf_y
        self.pf_mass = pf_mass 
        self.Wpnorms = Wpnorms

        self.update_count += 1
        
        # 리샘플링 조건 체크 (ESS)
        ess = 1.0 / np.sum(Wpnorms**2)
        if ess < self.pf_num * 0.5:
            self.update_count = 0
            self._particle_resample(likelihoods) # 인자 전달
        
        return self.pf_x, self.pf_y, self.pf_mass, self.Wpnorms
    
    # [추가] State 생성 시 NaN 방지를 위해 이 함수를 통해 값을 가져오세요
    def get_estimated_source(self):
        est_x = np.sum(self.pf_x * self.Wpnorms)
        est_y = np.sum(self.pf_y * self.Wpnorms)
        est_mass = np.sum(self.pf_mass * self.Wpnorms)
        
        # 만약 계산 결과가 NaN이면 맵 중앙값 리턴
        if np.isnan(est_x) or np.isnan(est_y):
            return self.court_lx/2, self.court_ly/2, self.min_cs_137_mass
            
        return est_x, est_y, est_mass

