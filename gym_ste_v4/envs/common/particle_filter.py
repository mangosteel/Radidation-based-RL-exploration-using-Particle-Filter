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


mode = 0 # 0 : 파라미터 k 고정  /  1: k 적응형 ....  11/28 기준 적응형이 run53 / 기존 모드가 run54 .. 검증성능으로 판단하자 일단 ㅇㅇ

# 일단 이걸로 학습시킨 후 temperature = 2.5 / k = * 0.09로 해서 검증함.. >> 아래의 적응형 코드의 성능이 별로라면 이걸로 학습을 진행하는걸로...
if mode == 0:
    import numpy as np
    import math
    from scipy.special import gammaln


    class ParticleFilter: 
        def __init__(self, args):
            self.update_count = -1
            self.pf_num = args.pf_num
            
            # [설정] 훈련/테스트 모드 구분 (args에 없으면 기본값 False)
            #self.is_test_mode = getattr(args, "test", False) or getattr(args, "test_mode", False)

            self.is_test_mode = True # 검증시에만 켜두고 평소에는 false

            # [설정] 환경 및 센서 노이즈 파라미터
            self.sensor_sig_m = args.sensor_sig_m 
            self.env_sig = args.env_sig        

            # [수정 1] LND 712 센서 스펙 적용 & 시간 간격
            self.measurement_interval = getattr(args, "delta_t", 1.0) 
            self.counts_per_uSv_per_h = getattr(args, "counts_per_uSv_per_h", 1.75)

            # [수정 2] Temperature 파라미터 분리 (Test 시 더 정밀하게)
            # 훈련 시: 15.0 (탐색 위주, 잘 안 죽음)
            # 테스트 시: 2.5 (정밀도 위주, 칼같이 평가)
            if self.is_test_mode:
                self.likelihood_temperature = 2.5  # 진짜 이거 때문에 파티클들이 과도하게 흩어져서 문제가 된듯... 수정된 파티클 필터 로직에 대해 공부하고 temperature 와 k에 대한 공부도 하자...
            else:
                self.likelihood_temperature = 15.0 

            self.radiation = args.radiation
            
            # 물리 상수
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

        def _pf_does_rate(self, agent_x, agent_y, source_x, source_y, source_mass): 
            """
            벡터화된(Numpy) 파티클 필터용 선량률 계산 함수
            """
            # [중요] 훈련/테스트 환경의 물리 상수 일치 (둘 다 12.0)
            # 만약 Test 코드에서 mu=124.19를 쓴다면 여기서 분기 처리 필요하지만,
            # 일반적으로는 물리 환경은 동일해야 함.
            mu = 12.0 
            
            # 1. 거리 제곱 계산
            dx = agent_x - source_x 
            dy = agent_y - source_y
            dist_sq = dx*dx + dy*dy 
            dist_sq = np.maximum(dist_sq, 1e-4) 
        
            # 2. 기본 선량률 계산
            dose_rate = self.cs_137_gamma * self.special_activity * source_mass / dist_sq
            
            # 3. 감쇠 적용 (Effective Source Model: 일괄 적용)
            barrier_thickness = 0.15 
            attenuation_factor = math.exp(-mu * barrier_thickness)
            dose_rate *= attenuation_factor
            
            return dose_rate
        
        def _dose_rate_to_cps(self, dose_rate_nSv):
            dose_rate_nSv = np.maximum(dose_rate_nSv, 0.0)
            dose_rate_uSv_per_h = dose_rate_nSv / 1000.0
            return dose_rate_uSv_per_h * self.counts_per_uSv_per_h

        def _get_lambda(self, expected_rate_nSv):
            signal_cps = self._dose_rate_to_cps(expected_rate_nSv)
            bg_cps = self._dose_rate_to_cps(self.env_sig) 
            total_lambda = (signal_cps + bg_cps) * self.measurement_interval
            return np.maximum(total_lambda, 1e-9) 

        def _poisson_likelihood(self, observation_nSv, expected_rate_nSv):
            obs_cps = self._dose_rate_to_cps(observation_nSv)
            k = np.round(obs_cps * self.measurement_interval).astype(int)
            
            lam = self._get_lambda(expected_rate_nSv)

            log_likelihood = k * np.log(lam) - lam - gammaln(k + 1)
            
            # Temperature 적용
            log_likelihood = log_likelihood / self.likelihood_temperature
            
            log_likelihood = np.nan_to_num(log_likelihood, nan=-1000.0, neginf=-1000.0)
            
            likelihood = np.exp(log_likelihood)
            likelihood = np.maximum(likelihood, 1e-100)
            
            return likelihood

        def _particle_resample(self, likelihoods):
            """
            [수정 3] Adaptive Jittering (적응형 노이즈) 적용
            파티클이 모이면 살살 흔들고, 퍼져있으면 세게 흔듭니다.
            """
            N = self.pf_num
            
            # 1. Systematic Resampling
            indx = np.zeros(N, dtype=int)
            Q = np.cumsum(self.Wpnorms)
            T = np.arange(N)/N + self.np_random.uniform(0, 1/N, N)
            
            i, j = 0, 0
            while i < N and j < N:
                while Q[j] < T[i]:
                    j += 1
                if j >= N: break
                indx[i] = j
                i += 1
                
            keep_x = self.pf_x[indx]
            keep_y = self.pf_y[indx]
            keep_mass = self.pf_mass[indx]
            
            # 2. Adaptive Jittering Parameter (K) 계산
            # 파티클들의 현재 표준편차(퍼짐 정도)를 계산
            std_x = np.std(self.pf_x)
            std_y = np.std(self.pf_y)
            avg_std = (std_x + std_y) / 2.0
            
            # 맵 크기 대비 퍼짐 비율 (0.0 ~ 1.0)
            spread_ratio = avg_std / max(self.court_lx, 1.0)
            
            # 퍼짐 비율에 비례하여 K값 자동 조절
            # 많이 퍼짐(ratio=0.5) -> K=0.05 (5% 흔들기)
            # 많이 모임(ratio=0.01) -> K=0.005 (0.5% 흔들기)
            # 최소 0.5% ~ 최대 5% 사이로 제한
            K = np.clip(spread_ratio * 0.1, 0.005, 0.05)
            
            # Test 모드에서는 마지막 수렴을 위해 노이즈를 더 줄임 (선택 사항)
            if self.is_test_mode:
                K *= 0.09 # 절반으로 더 줄임

            # 3. Roughening (Noise Injection)
            jitter_x = K * self.court_lx * self.np_random.normal(0, 1, N)
            jitter_y = K * self.court_ly * self.np_random.normal(0, 1, N)
            
            mass_range = self.max_cs_137_mass - self.min_cs_137_mass
            jitter_mass = K * mass_range * self.np_random.normal(0, 1, N)

            # 4. 업데이트 및 Clip
            self.pf_x = np.clip(keep_x + jitter_x, 0, self.court_lx)
            self.pf_y = np.clip(keep_y + jitter_y, 0, self.court_ly)
            self.pf_mass = np.clip(keep_mass + jitter_mass, self.min_cs_137_mass, self.max_cs_137_mass)

            # 5. 가중치 초기화
            self.Wpnorms = np.ones(N) / N

        def _weight_calculate(self, radiation_measure, agent_x, agent_y, pf_x, pf_y, pf_mass):
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
            Wps = Wpnorms * (likelihoods**(1/num_connected))
            
            Wp_sum = np.sum(Wps)
            if Wp_sum == 0 or np.isnan(Wp_sum):
                Wpnorms = np.ones(self.pf_num) / self.pf_num
                resample_true = True
            else:
                Wpnorms = Wps / Wp_sum
                
                # 3. 리샘플링 조건 (Effective Sample Size)
                n_eff = 1.0 / np.sum(Wpnorms**2)
                
                # [수정 4] 리샘플링 임계값 조정 (Test 모드 시 더 자주 리샘플링하지 않도록)
                # 파티클의 50%가 죽었을 때만 리샘플링
                resample_true = n_eff < (self.pf_num * 0.5)

            self.pf_x = pf_x
            self.pf_y = pf_y
            self.Wpnorms = Wpnorms
            self.update_count += 1

            if resample_true: 
                self.update_count = 0
                self._particle_resample(likelihoods) 

            return self.pf_x, self.pf_y, self.pf_mass, self.Wpnorms


# 이건 파티클 분포 적응형 ver >> 아직 정확히는 모르지만 파티클 분포에 따라 K가 adaptive하게 변하는 듯...
elif mode ==1:
    import numpy as np
    import math
    from scipy.special import gammaln

    class ParticleFilter: 
        def __init__(self, args):
            self.update_count = -1
            self.pf_num = args.pf_num
            
            # [설정] 테스트 모드 확인 (args에 test_mode가 없으면 기본값 False)
            # 훈련 코드에서는 False, 테스트 코드에서는 True로 설정해서 넘겨주세요.
            #self.is_test_mode = getattr(args, "test_mode", False) or getattr(args, "test", False)

            self.is_test_mode = False  # 현재는 학습모드!
            # [설정] 환경 및 센서 노이즈 파라미터
            self.sensor_sig_m = args.sensor_sig_m 
            self.env_sig = args.env_sig        

            self.measurement_interval = getattr(args, "delta_t", 1.0) 
            self.counts_per_uSv_per_h = getattr(args, "counts_per_uSv_per_h", 1.75)

            # [전략 1] Temperature 자동 조절
            # 훈련 시: 15.0 (파티클 생존 우선 -> RL 학습 안정성 확보)
            # 테스트 시: 2.5 (정밀도 우선 -> 정확한 위치 추정)
            if self.is_test_mode:
                self.likelihood_temperature = 2.5
            else:
                self.likelihood_temperature = 15.0 

            self.radiation = args.radiation
            
            # 물리 상수
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
            """
            벡터화된(Numpy) 파티클 필터용 선량률 계산 함수
            """
            # [중요] 물리 환경 통일: mu = 12.0 (콘크리트/흙벽 수준)
            # 신호가 벽을 뚫고 나와야 파티클이 '기울기'를 보고 찾아갑니다.
            mu = 12.0 
            
            # 1. 거리 제곱 계산
            dx = agent_x - source_x 
            dy = agent_y - source_y
            dist_sq = dx*dx + dy*dy 
            dist_sq = np.maximum(dist_sq, 1e-4) 
        
            # 2. 기본 선량률 계산
            dose_rate = self.cs_137_gamma * self.special_activity * source_mass / dist_sq
            
            # 3. 감쇠 적용 (Effective Source Model)
            # 벽 밖의 파티클도 일괄 감쇠하지만, 거리 역제곱 법칙에 의해 벽 밖 파티클은 
            # 값이 폭발(Spike)하므로 필터링 과정에서 자연스럽게 제거됩니다.
            barrier_thickness = 0.15 
            attenuation_factor = math.exp(-mu * barrier_thickness)
            dose_rate *= attenuation_factor
            
            return dose_rate
        
        def _dose_rate_to_cps(self, dose_rate_nSv):
            dose_rate_nSv = np.maximum(dose_rate_nSv, 0.0)
            dose_rate_uSv_per_h = dose_rate_nSv / 1000.0
            return dose_rate_uSv_per_h * self.counts_per_uSv_per_h

        def _get_lambda(self, expected_rate_nSv):
            signal_cps = self._dose_rate_to_cps(expected_rate_nSv)
            bg_cps = self._dose_rate_to_cps(self.env_sig) 
            total_lambda = (signal_cps + bg_cps) * self.measurement_interval
            return np.maximum(total_lambda, 1e-9) 

        def _poisson_likelihood(self, observation_nSv, expected_rate_nSv):
            obs_cps = self._dose_rate_to_cps(observation_nSv)
            k = np.round(obs_cps * self.measurement_interval).astype(int)
            
            lam = self._get_lambda(expected_rate_nSv)

            log_likelihood = k * np.log(lam) - lam - gammaln(k + 1)
            
            # Temperature Scaling (생존율 조절)
            log_likelihood = log_likelihood / self.likelihood_temperature
            
            # 수치 안정성
            log_likelihood = np.nan_to_num(log_likelihood, nan=-1000.0, neginf=-1000.0)
            likelihood = np.exp(log_likelihood)
            likelihood = np.maximum(likelihood, 1e-100)
            
            return likelihood

        def _particle_resample(self, likelihoods):
            """
            [전략 2] Adaptive Jittering (적응형 노이즈)
            파티클이 퍼져있으면(분산 큼) -> 세게 흔들어서 탐색 (High K)
            파티클이 모여있으면(분산 작음) -> 살살 흔들어서 수렴 (Low K)
            """
            N = self.pf_num
            
            # 1. Systematic Resampling
            indx = np.zeros(N, dtype=int)
            Q = np.cumsum(self.Wpnorms)
            T = np.arange(N)/N + self.np_random.uniform(0, 1/N, N)
            
            i, j = 0, 0
            while i < N and j < N:
                while Q[j] < T[i]:
                    j += 1
                if j >= N: break
                indx[i] = j
                i += 1
                
            keep_x = self.pf_x[indx]
            keep_y = self.pf_y[indx]
            keep_mass = self.pf_mass[indx]
            
            # 2. Adaptive K 값 계산 (핵심!)
            # 파티클들의 현재 퍼짐 정도(Standard Deviation) 계산
            std_x = np.std(self.pf_x)
            std_y = np.std(self.pf_y)
            avg_std = (std_x + std_y) / 2.0
            
            # 맵 크기(60m) 대비 파티클이 얼마나 퍼져있는지 비율 (0.0 ~ 1.0)
            spread_ratio = avg_std / max(self.court_lx, 1.0)
            
            # 퍼짐 비율에 따라 K값을 동적으로 조절
            # spread_ratio가 0.2(20%)면 -> K=0.04
            # spread_ratio가 0.01(1%)면 -> K=0.002
            K = np.clip(spread_ratio * 0.2, 0.002, 0.05)
            
            # 테스트 모드라면 더더욱 정밀하게(절반으로 줄임)
            if self.is_test_mode:
                K *= 0.09 

            # 3. Roughening (Noise Injection)
            jitter_x = K * self.court_lx * self.np_random.normal(0, 1, N)
            jitter_y = K * self.court_ly * self.np_random.normal(0, 1, N)
            
            mass_range = self.max_cs_137_mass - self.min_cs_137_mass
            jitter_mass = K * mass_range * self.np_random.normal(0, 1, N)

            # 4. 업데이트 및 Clip (맵 밖으로 나가는 것 방지)
            self.pf_x = np.clip(keep_x + jitter_x, 0, self.court_lx)
            self.pf_y = np.clip(keep_y + jitter_y, 0, self.court_ly)
            self.pf_mass = np.clip(keep_mass + jitter_mass, self.min_cs_137_mass, self.max_cs_137_mass)

            # 5. 가중치 초기화
            self.Wpnorms = np.ones(N) / N

        def _weight_calculate(self, radiation_measure, agent_x, agent_y, pf_x, pf_y, pf_mass):
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
            Wps = Wpnorms * (likelihoods**(1/num_connected))
            
            Wp_sum = np.sum(Wps)
            if Wp_sum == 0 or np.isnan(Wp_sum):
                Wpnorms = np.ones(self.pf_num) / self.pf_num
                resample_true = True
            else:
                Wpnorms = Wps / Wp_sum
                
                # 3. 리샘플링 조건 (Effective Sample Size)
                n_eff = 1.0 / np.sum(Wpnorms**2)
                
                # 훈련 시엔 50% 미만일 때, 테스트 시엔 30% 미만일 때만 리샘플링 (테스트 시엔 최대한 정보 유지)
                threshold = 0.3 if self.is_test_mode else 0.5
                resample_true = n_eff < (self.pf_num * threshold)

            self.pf_x = pf_x
            self.pf_y = pf_y
            self.Wpnorms = Wpnorms
            self.update_count += 1

            if resample_true: 
                self.update_count = 0
                self._particle_resample(likelihoods) 

            return self.pf_x, self.pf_y, self.pf_mass, self.Wpnorms