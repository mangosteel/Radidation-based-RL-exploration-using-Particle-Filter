# import numpy as np
# import math
# from scipy.special import gammaln

# class ParticleFilter: 
#     def __init__(self, args):
#         self.update_count = -1
#         self.sensor_sig_m = args.sensor_sig_m 
#         self.env_sig = args.env_sig        
#         self.pf_num = args.pf_num    

#         # delta_t: 카운트 계산을 위한 시간 간격
#         self.measurement_interval = getattr(args, "delta_t", 1.0) 
#         self.counts_per_uSv_per_h = getattr(args, "counts_per_uSv_per_h", 2.0)

#         self.radiation = args.radiation
        
#         self.max_cs_137_mass = 9.47 * 10**(-6) 
#         self.min_cs_137_mass = self.max_cs_137_mass / 2 
#         self.special_activity = 3.2*10**12 
#         self.cs_137_gamma = 330/(1000) # 아마도 nSv/h로 나올듯?
#         self.court_lx = args.court_lx   
#         self.court_ly = args.court_ly

#         self.np_random = args.np_random 

#         # 파티클 초기화
#         pf_low_state_x = np.zeros(self.pf_num) 
#         pf_low_state_y = np.zeros(self.pf_num) 
#         pf_low_mass = np.ones(self.pf_num) * self.min_cs_137_mass
        
#         pf_high_state_x = np.ones(self.pf_num)*self.court_lx 
#         pf_high_state_y = np.ones(self.pf_num)*self.court_ly
#         pf_high_mass = np.ones(self.pf_num)*self.max_cs_137_mass 
        
#         self.pf_x = self.np_random.uniform(low = pf_low_state_x, high = pf_high_state_x) 
#         self.pf_y = self.np_random.uniform(low = pf_low_state_y, high = pf_high_state_y)
#         self.pf_mass = self.np_random.uniform(low = pf_low_mass, high = pf_high_mass) 
        
#         self.Wpnorms = np.ones(self.pf_num)/self.pf_num 

#     def _pf_does_rate(self, agent_x, agent_y, source_x, source_y, source_mass): # 정확한 파티클 측정치 감쇠 로직 구현이 필요!!!
#         """
#         벡터화된(Numpy) 파티클 필터용 선량률 계산 함수
#         source_x, source_y, source_mass는 모두 (pf_num,) 크기의 배열입니다.
#         """
#         mu = 124.19 # 감쇠 계수
        
#         # 1. 거리 제곱 계산 (r^2)
#         dx = agent_x - source_x 
#         dy = agent_y - source_y
#         dist_sq = dx*dx + dy*dy # r은 여기서 거리의 제곱입니다!
        
#         # 0으로 나누기 방지 (최소 거리 제곱 1cm^2)
#         dist_sq = np.maximum(dist_sq, 1e-4) 
    
#         # 2. 기본 선량률 계산 (거리 역제곱 법칙)
#         # 결과값 dose_rate는 파티클 개수만큼의 배열이 됩니다.
#         dose_rate = self.cs_137_gamma * self.special_activity * source_mass / dist_sq
        
#         # 3. 장애물(원형 차폐막) 감쇠 적용 준비
#         barrier_radius = 1.75
#         barrier_thickness = 0.15 
        
#         # # 비교 기준 거리 설정 (반경 + 두께/2)
#         # threshold_dist = barrier_radius + barrier_thickness / 2.0   # 생각해보니 어차피 모든 파티클에는 감쇠가 적용되어야 하는데 굳이 조건을 줄필요 없음.. 이렇게 되면 가까운 파티클은 제 값을 받아버림...
        
#         # # [중요] r이 '거리 제곱'이므로, 비교할 기준값도 '제곱'해줘야 단위가 맞습니다.
#         # threshold_sq = threshold_dist ** 2 

#         # 감쇠 계수 계산 (상수)
#         attenuation_factor = math.exp(-mu * barrier_thickness)

#         # 4. Numpy 마스킹을 이용한 조건부 감쇠 적용
#         # "거리가 기준보다 먼 경우"에 해당하는 인덱스(True/False)를 찾습니다.
#         # 즉, 벽을 뚫고 나온 방사선(Inside -> Outside)인 경우입니다.
#         # mask = dist_sq > threshold_sq 
        
#         # mask가 True인 위치의 dose_rate에만 감쇠 계수를 곱합니다.
#         dose_rate *= attenuation_factor
        
#         return dose_rate
    
#     def _dose_rate_to_cps(self, dose_rate_nSv):
#         # nSv/h -> uSv/h -> CPS 변환
#         dose_rate_nSv = np.maximum(dose_rate_nSv, 0.0)
#         dose_rate_uSv_per_h = dose_rate_nSv / 1000.0
#         return dose_rate_uSv_per_h * self.counts_per_uSv_per_h

#     def _get_lambda(self, expected_rate_nSv):
#         # 파티클 예측 선량률 -> 기대 카운트(Lambda)
#         # 1. 신호 성분
#         signal_cps = self._dose_rate_to_cps(expected_rate_nSv)
#         # 2. 배경 성분 (env_sig를 배경잡음 nSv/h로 가정)
#         bg_cps = self._dose_rate_to_cps(self.env_sig) 
        
#         # 총 기대 카운트 (Lambda)
#         total_lambda = (signal_cps + bg_cps) * self.measurement_interval
#         return np.maximum(total_lambda, 1e-9) # 로그 계산 오류 방지용 최소값

#     def _poisson_likelihood(self, observation_nSv, expected_rate_nSv):
#         # 1. 실제 관측값(nSv)을 정수형 카운트(k)로 변환
#         obs_cps = self._dose_rate_to_cps(observation_nSv)
#         k = int(np.round(obs_cps * self.measurement_interval) )
        
#         # 2. 파티클 예측값(nSv)을 Lambda로 변환
#         lam = self._get_lambda(expected_rate_nSv)

#         # 3. Log-Likelihood 계산: k*ln(lam) - lam - ln(k!)
#         log_likelihood = k * np.log(lam) - lam - gammaln(k + 1)
        
#         # NaN 및 무한대 처리 (매우 중요!)
#         log_likelihood = np.nan_to_num(log_likelihood, nan=-100.0, posinf=-100.0, neginf=-100.0)
        
#         # exp 처리 시 Underflow 방지
#         likelihood = np.exp(log_likelihood)
#         likelihood = np.nan_to_num(likelihood, nan=1e-30) # NaN 방지
        
#         return likelihood

#     def _weight_calculate(self, radiation_measure, agent_x, agent_y, pf_x, pf_y, pf_mass):
#         # 외부에서 호출 시 사용 (구조 유지)
#         pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass) # 에이전트와 근원지 사이의 측정값...
#         poisson_weight = self._poisson_likelihood(radiation_measure, pf_dose_rate)
#         poisson_weight[poisson_weight != poisson_weight] = 1e-200 # nan이 되면 자기자신과 달라짐 그래서 nan방지하는 거임!
#         poisson_weight[poisson_weight < 1e-200] = 1e-200
#         return poisson_weight

#     def _particle_resample(self, likelihoods):
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
#                 alpha = n_new/likelihoods[indx] # 새롭게 정의된 가중치와 이전의 중첩을 포함한 가중치간의 비율..
#                 mcrand = self.np_random.uniform(0,1,self.pf_num) # 파티클 개수만큼의 0~1의 난수.
# #                print(alpha > mcrand)
#                 new_point_bool = alpha > mcrand # 어차피 스무딩 가중치가 이전보다 좋다면 1이 넘을테니 thersold가 충분히 됨!
#                 self.pf_x[new_point_bool] = nXxp[new_point_bool] # 1보다 작다면 어차피 원래꺼랑 비슷해서 아무거나 써도 상관없음!
#                 self.pf_y[new_point_bool] = nXyp[new_point_bool]
#                 self.pf_mass[new_point_bool] = nX_mass_xp[new_point_bool] # 아 애초에 n_new로 pf를 업데이트 시키는구나...
#             self.Wpnorms = np.ones(self.pf_num)/self.pf_num # 아 그래서 다시 초기화 시키는 건가? 이미 리샘플링 할때 가중치 써먹었으니...

#     def _weight_update(self, measure, agent_x, agent_y, pf_x, pf_y , pf_mass, Wpnorms, num_connected):
#         Wp_sum = 0
#         resample_true = False

#         self.radiation_measure = measure 
#         self.agent_x = agent_x
#         self.agent_y = agent_y

#         # 1. 우도 계산
#         pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass)
#         likelihoods = self._poisson_likelihood(self.radiation_measure, pf_dose_rate)

#         likelihoods[likelihoods != likelihoods] = 1e-200
#         likelihoods[likelihoods < 1e-200] = 1e-200
#         sort_g = np.sort(likelihoods) # ort_g = np.sort(gauss_new)로 우도 값을 정렬하여, 모든 파티클의 우도 값이 동일한지 확인
#         if (sort_g[self.pf_num-1] == sort_g[0]): resample_true = True # 가장 큰 값과 작은 값이 동일하다면 이는 모든 값이 같으므로...리샘플링!
#         #if (self.update_count == 10): resample_true = True
#         Wps = Wpnorms * (likelihoods**(1/num_connected))
#         Wp_sum = np.sum(Wps) 

#         Wpnorms = Wps/Wp_sum 


#         self.pf_x = pf_x # 가중치 업데이트 시 정의된 입력 파티클의 위치 및 방출강도!
#         self.pf_y = pf_y # 근데 이건 초기의 입력값 아닌가? 다시 초기화 되는거 아님? 이것의 의도가 뭘까? / 교수님께 질문해보자...
       
#         self.Wpnorms = Wpnorms

#         self.update_count += 1

#         if 1/sum(pow(Wpnorms,2)) < self.pf_num*0.5 or resample_true: # 1 for every time
#             self.update_count = 0 # 이건 리샘플링을 했다는 증거이다!
#             self._particle_resample(likelihoods) # self.pf_x, self.pf_y, self.pf_source_activity, self.Wpnorms이 업데이트...
#             #self.CovXxp = np.var(self.pf_x)
#             #self.CovXyp = np.var(self.pf_y)
#             #self.CovX_radi_xp = np.var(self.pf_source_activity)
#             #self.resampled = 1


#         return self.pf_x, self.pf_y, self.pf_mass, self.Wpnorms # 

#수정 ver 
import numpy as np
import math
from scipy.special import gammaln

class ParticleFilter: 
    def __init__(self, args):
        self.update_count = -1
        self.pf_num = args.pf_num
        
        # [설정] 환경 및 센서 노이즈 파라미터
        self.sensor_sig_m = args.sensor_sig_m 
        self.env_sig = args.env_sig        

        # [수정 1] LND 712 센서 스펙 적용 & 시간 간격
        self.measurement_interval = getattr(args, "delta_t", 1.0) 
        # LND 712 감도: 약 1.75 CPS per uSv/h (상용 센서 표준)
        self.counts_per_uSv_per_h = getattr(args, "counts_per_uSv_per_h", 1.75)

        # [수정 2] 수렴 개선을 위한 Temperature 파라미터 (중요!)
        # 1.0이면 정석 포아송 분포. 값을 키울수록(5~10) 분포가 완만해져서 수렴이 잘 됨.
        self.likelihood_temperature = 5.0 

        self.radiation = args.radiation
        
        # 물리 상수 및 소스 파라미터
        self.max_cs_137_mass = 9.47 * 10**(-6) 
        self.min_cs_137_mass = self.max_cs_137_mass / 2 
        self.special_activity = 3.2*10**12 
        self.cs_137_gamma = 330/(1000) 
        self.court_lx = args.court_lx   
        self.court_ly = args.court_ly

        self.np_random = args.np_random 

        # 파티클 초기화 (Uniform Distribution)
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

    def _pf_does_rate(self, agent_x, agent_y, source_x, source_y, source_mass): # 정확한 파티클 측정치 감쇠 로직 구현이 필요!!!
        """
        벡터화된(Numpy) 파티클 필터용 선량률 계산 함수
        source_x, source_y, source_mass는 모두 (pf_num,) 크기의 배열입니다.
        """
        mu = 12.419 # 감쇠 계수
        
        # 1. 거리 제곱 계산 (r^2)
        dx = agent_x - source_x 
        dy = agent_y - source_y
        dist_sq = dx*dx + dy*dy # r은 여기서 거리의 제곱입니다!
        
        # 0으로 나누기 방지 (최소 거리 제곱 1cm^2)
        dist_sq = np.maximum(dist_sq, 1e-4) 
    
        # 2. 기본 선량률 계산 (거리 역제곱 법칙)
        # 결과값 dose_rate는 파티클 개수만큼의 배열이 됩니다.
        dose_rate = self.cs_137_gamma * self.special_activity * source_mass / dist_sq
        
        # 3. 장애물(원형 차폐막) 감쇠 적용 준비
        barrier_radius = 1.75
        barrier_thickness = 0.15 
        
        # 비교 기준 거리 설정 (반경 + 두께/2)
        # threshold_dist = barrier_radius + barrier_thickness / 2.0
        
        # # [중요] r이 '거리 제곱'이므로, 비교할 기준값도 '제곱'해줘야 단위가 맞습니다.
        # threshold_sq = threshold_dist ** 2 

        # 감쇠 계수 계산 (상수)
        attenuation_factor = math.exp(-mu * barrier_thickness )

        # 4. Numpy 마스킹을 이용한 조건부 감쇠 적용
        # "거리가 기준보다 먼 경우"에 해당하는 인덱스(True/False)를 찾습니다.
        # 즉, 벽을 뚫고 나온 방사선(Inside -> Outside)인 경우입니다.
        #mask = dist_sq > threshold_sq 
        
        # mask가 True인 위치의 dose_rate에만 감쇠 계수를 곱합니다.
        dose_rate *= attenuation_factor
        
        return dose_rate
    
    def _dose_rate_to_cps(self, dose_rate_nSv):
        # nSv/h -> uSv/h -> CPS 변환
        dose_rate_nSv = np.maximum(dose_rate_nSv, 0.0)
        dose_rate_uSv_per_h = dose_rate_nSv / 1000.0
        return dose_rate_uSv_per_h * self.counts_per_uSv_per_h

    def _get_lambda(self, expected_rate_nSv):
        # 파티클 예측 선량률 -> 기대 카운트(Lambda)
        signal_cps = self._dose_rate_to_cps(expected_rate_nSv)
        bg_cps = self._dose_rate_to_cps(self.env_sig) 
        
        total_lambda = (signal_cps + bg_cps) * self.measurement_interval
        # 로그 계산 오류 방지를 위한 최소값 설정
        return np.maximum(total_lambda, 1e-9) 

    def _poisson_likelihood(self, observation_nSv, expected_rate_nSv):
        # [수정 3] 포아송 우도 함수 개선
        
        # 1. 실제 관측값(nSv) -> 정수형 카운트(k) (확실한 int 변환)
        obs_cps = self._dose_rate_to_cps(observation_nSv)
        k = np.round(obs_cps * self.measurement_interval).astype(int)
        
        # 2. 예측값 -> Lambda
        lam = self._get_lambda(expected_rate_nSv)

        # 3. Log-Likelihood: k*ln(lam) - lam - ln(k!)
        log_likelihood = k * np.log(lam) - lam - gammaln(k + 1)
        
        # [중요] Temperature 적용 (분포 물타기)
        log_likelihood = log_likelihood / self.likelihood_temperature
        
        # 4. 수치 안정성 처리 (NaN, Inf 방지)
        log_likelihood = np.nan_to_num(log_likelihood, nan=-1000.0, neginf=-1000.0)
        
        # 5. exp 변환 및 Underflow 방지
        likelihood = np.exp(log_likelihood)
        likelihood = np.maximum(likelihood, 1e-100) # 0이 되는 것 방지
        
        return likelihood

    def _particle_resample(self, likelihoods):
        """
        [수정 4] MCMC 제거 -> Systematic Resampling + Roughening (Jittering)
        포아송 모델처럼 뾰족한 분포에서는 이 방식이 훨씬 강건합니다.
        """
        N = self.pf_num
        
        # 1. Systematic Resampling (기존 로직 유지)
        indx = np.zeros(N, dtype=int)
        Q = np.cumsum(self.Wpnorms)
        T = np.arange(N)/N + self.np_random.uniform(0, 1/N, N)
        
        i, j = 0, 0
        while i < N and j < N:
            while Q[j] < T[i]:
                j += 1
            if j >= N: break # 인덱스 초과 방지
            indx[i] = j
            i += 1
            
        # 선택된 우수 파티클 추출
        keep_x = self.pf_x[indx]
        keep_y = self.pf_y[indx]
        keep_mass = self.pf_mass[indx]
        
        # 2. Roughening (파티클 뭉침 방지 - 강제 노이즈 주입)
        # 튜닝 파라미터 K: 전체 맵 크기의 2% 정도 흔들어줌 (수렴 속도 조절)
        K = 0.02 
        
        # 위치 노이즈
        jitter_x = K * self.court_lx * self.np_random.normal(0, 1, N)
        jitter_y = K * self.court_ly * self.np_random.normal(0, 1, N)
        
        # 질량 노이즈
        mass_range = self.max_cs_137_mass - self.min_cs_137_mass
        jitter_mass = K * mass_range * self.np_random.normal(0, 1, N)

        # 3. 파티클 업데이트 및 경계값 처리 (Clip)
        self.pf_x = np.clip(keep_x + jitter_x, 0, self.court_lx)
        self.pf_y = np.clip(keep_y + jitter_y, 0, self.court_ly)
        self.pf_mass = np.clip(keep_mass + jitter_mass, self.min_cs_137_mass, self.max_cs_137_mass)

        # 4. 가중치 초기화 (1/N)
        self.Wpnorms = np.ones(N) / N

    def _weight_calculate(self, radiation_measure, agent_x, agent_y, pf_x, pf_y, pf_mass):
        # 외부 호출용 (구조 유지)
        pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass)
        poisson_weight = self._poisson_likelihood(radiation_measure, pf_dose_rate)
        return poisson_weight

    def _weight_update(self, measure, agent_x, agent_y, pf_x, pf_y , pf_mass, Wpnorms, num_connected):
        self.radiation_measure = measure 
        self.agent_x = agent_x
        self.agent_y = agent_y

        # 1. 우도 계산
        pf_dose_rate = self._pf_does_rate(agent_x, agent_y, pf_x, pf_y, pf_mass)
        likelihoods = self._poisson_likelihood(self.radiation_measure, pf_dose_rate)

        # 2. 가중치 업데이트
        # (num_connected는 보통 1을 사용하나, 기존 로직 존중)
        Wps = Wpnorms * (likelihoods**(1/num_connected))
        
        # NaN 방지 및 정규화
        Wp_sum = np.sum(Wps)
        if Wp_sum == 0 or np.isnan(Wp_sum):
            # 모든 파티클이 죽은 경우 -> 랜덤 리셋 or 균등 분배
            Wpnorms = np.ones(self.pf_num) / self.pf_num
            resample_true = True
        else:
            Wpnorms = Wps / Wp_sum
            
            # 유효 파티클 개수(N_eff) 계산
            n_eff = 1.0 / np.sum(Wpnorms**2)
            # 파티클 절반 이하만 유효하면 리샘플링
            resample_true = n_eff < (self.pf_num * 0.5)

        self.pf_x = pf_x
        self.pf_y = pf_y
        self.Wpnorms = Wpnorms
        self.update_count += 1

        # 3. 리샘플링 실행
        # (혹은 일정 주기마다 강제 리샘플링)
        if resample_true: 
            self.update_count = 0
            self._particle_resample(likelihoods) 

        return self.pf_x, self.pf_y, self.pf_mass, self.Wpnorms