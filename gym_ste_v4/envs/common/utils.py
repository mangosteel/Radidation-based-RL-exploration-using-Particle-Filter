import math
import numpy as np
from itertools import chain

debug_on = False

def d_btw_points(p_1, p_2):
    return math.sqrt( (p_1[0]-p_2[0])**2 + (p_1[1]-p_2[1])**2 )

def line_intersects(x1, y1, x2, y2, x3, y3, x4, y4):
    """두 선분 (x1,y1)-(x2,y2)와 (x3,y3)-(x4,y4)가 교차하는지 판정"""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    A, B, C, D = (x1,y1), (x2,y2), (x3,y3), (x4,y4)
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def segment_intersection(p1, p2, q1, q2):
    """Return intersection point and ratio along p1->p2 segment if it exists."""

    def cross(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    denom = cross(r, s)
    if abs(denom) < 1e-12:
        return None

    qp = (q1[0] - p1[0], q1[1] - p1[1])
    t = cross(qp, s) / denom
    u = cross(qp, r) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * r[0], p1[1] + t * r[1]), t
    return None




def radiation_field(x, y, radiation, obstacles=None,visual=True): # 이것을 시각화용 실제 값 전용 두가지 버전을 만들어야 할듯..
    """
    방사능 선량률 계산 (거리 기반 + 장애물 감쇠 반영)

    Parameters
    ----------
    x, y : float
        측정 위치 (m 단위)
    radiation : Radiation
        S_x, S_y, cs_137_gamma, S_mass, special_activity 보유 객체
    obstacles : list of ((x1,y1),(x2,y2)) or ((x1,y1),(x2,y2),thickness)
        장애물 직선의 양 끝점 (and Optional: 두께[m])
    mu : float
        감쇠 계수 [m⁻¹] (납벽: 1.2419 cm⁻¹ = 124.19 m⁻¹)
    """
    if visual:
        mu=12.419
    else:
        mu = 124.19
    # 1) 기본 dose (거리 기반)
    dx = x - radiation.S_x
    dy = y - radiation.S_y
    r = math.hypot(dx, dy)
    r = max(r, 1e-6)
    dose = radiation.cs_137_gamma * radiation.S_mass * radiation.special_activity / (r**2)

    # 2) 장애물 감쇠 ∏ e^(–μ · d_t)
    if obstacles:
        attenuation = 1.0
        for obs in obstacles:
            # obs 가 2-tuple 이면 thickness=1m, 3-tuple 이면 내부 값 사용
            if len(obs) == 2:
                (x1, y1), (x2, y2) = obs
                thickness = 0.04 # 1.0
            elif len(obs) == 3:
                (x1, y1), (x2, y2), thickness = obs
            else:
                continue

            # 소스-측정 지점 선분이 벽 선분과 교차하면 attenuation 적용
            if line_intersects(x1, y1, x2, y2,
                               radiation.S_x, radiation.S_y, x, y):
                attenuation *= math.exp(-mu * thickness)
        dose *= attenuation

    return dose


def radiation_field_for_render(x, y, raidation): # 방사능 농도  / 어차피 radiation에 방사선 객체가 들어감!
    q= 2000
    # Calculate the Euclidean distance from the point (x, y) to the source (S_x, S_y).
    dx = x - raidation.S_x  # raidation.S_x만 env에 따로 정의해놓으면댐 가스께 복붙해서...
    dy = y - raidation.S_y
    r = np.sqrt(dx*dx + dy*dy)  # 넘파이 값은 np.sqrt로 해야댐! TypeError: only size-1 arrays can be converted to Python scalars 아니면 이거뜸!

    
    # To avoid division by zero in case the detection point is extremely close to the source:
    # if r < 1e-6:
    #     r = 1e-6
    r = np.maximum(r, 1e-6) # 

    # Free-space radiation dose_rate rate model: dose_rate = q / r^2
    dose_rate = q / (r ** 2)  # 어차피 값은 가스꺼 그대로 쓰고 변수명만 슬쩍 바꾸면댐!
    
    return dose_rate



def isotropic_plume(pos_x, pos_y, gas, t_wind): # true gas conectration 가스농도 계산함수! / 노이즈가 없는 가스농도 계산. 이는 후에 우도함수의 평균으로 사용된다. 그리고 센서 노이즈의 분산으로 사용된다/
    if gas.S_x == pos_x and gas.S_y == pos_y: # to avoid divide by 0 / gas , wind객체를 의미하는듯 그래서 특성값들을 받을 수 있는듯...
        pos_x += 1e-10  # 소스와 에이전트위치가 동일하다면 약간의 차이(사실상0에 가까운)를 둔다.
        pos_y += 1e-10
    dist = math.sqrt(pow((gas.S_x - pos_x), 2) + pow(gas.S_y - pos_y, 2))  
    y_n = -(pos_x - gas.S_x)*math.sin(t_wind.mean_phi*math.pi/180) \
          +(pos_y - gas.S_y)*math.cos(t_wind.mean_phi*math.pi/180)
    lambda_plume = math.sqrt(gas.d * gas.t / (1 + pow(t_wind.mean_speed,2) * gas.t/4/gas.d) )
    
    conc = gas.q/(4 * math.pi * gas.d * dist) * np.exp(-y_n * t_wind.mean_speed/(2*gas.d) - dist/lambda_plume)
    if conc < 0:  # conc가 노이즈 없는 평균 가스 농도/ 논문에서는 m(p|seta)로 표기함..
        conc = 0
    return conc #- self.conc_eps


def crash_check(pre_pose_uav, num_agents, uav, crash_range, crash_warning_range, step_size, delta_t):
    # Calculate distance btw agents in contiuous time domain using velocity
    vel = []
    root_data = np.zeros([num_agents, num_agents])
    for n in range(num_agents):
        vel.append([uav[n].x-pre_pose_uav[n][0],
                    uav[n].y-pre_pose_uav[n][1] ] )

    crash_list = np.zeros([num_agents, num_agents], dtype=bool)
    crash_warning_list = np.zeros([num_agents, num_agents], dtype=bool)

    for n in range(num_agents):
        for m in range(n+1, num_agents):
            root_1 = root_2 = -1e5
            uav_m = uav[m]
            uav_n = uav[n]
            m_n_dist_pre = d_btw_points(pre_pose_uav[m],pre_pose_uav[n])
            m_n_dist     = d_btw_points([uav_m.x, uav_m.y], [uav_n.x, uav_n.y])
            if debug_on:
                print("Agent ", [n, m], "th before position: ", [pre_pose_uav[n], pre_pose_uav[m]])
                print("Distance before move: ", m_n_dist_pre)
                print("Distance after move: ", m_n_dist)
            if m_n_dist < crash_warning_range:
                crash_warning_list[n][m] = True
                crash_warning_list[m][n] = True
            if m_n_dist_pre < step_size*2 + crash_range:
                delta_p = np.array([pre_pose_uav[m][0]-pre_pose_uav[n][0],
                                    pre_pose_uav[m][1]-pre_pose_uav[n][1]])
                delta_v = np.array([vel[m][0]-vel[n][0],
                                    vel[m][1]-vel[n][1] ])

                a_const = np.dot(delta_v, delta_v)
                b_const = 2*np.dot(delta_v, delta_p)
                c_const = np.dot(delta_p, delta_p) - crash_range**2
                if debug_on: print("A_CONST: ", a_const, "   B_CONST: ", b_const, "  C_CONST: ", c_const)
                if c_const< 0:
                    crash_list[n][m] = True
                    crash_list[m][n] = True
                    continue

                b_sq_4ac = b_const**2 - 4*a_const*c_const
                root_data[n][m] = b_sq_4ac
                if b_sq_4ac > 0:
                    root_1 = (-b_const + math.sqrt(b_sq_4ac) )/(2*a_const)
                    root_2 = (-b_const - math.sqrt(b_sq_4ac) )/(2*a_const)
                    if debug_on:
                        print("ROOT 1 : ", root_1, "    ROOT 2 : ", root_2)
                        print("Agent ", [n, m], "th after position: ", [[uav_n.x, uav_n.y], [uav_m.x, uav_m.y]])
                        print("Distance after move: ", m_n_dist)
                    if root_1 > 0 and root_1 < delta_t:
                        crash_list[n][m] = True
                        crash_list[m][n] = True
                        continue
                    elif root_2 > 0 and root_2 < delta_t:
                        crash_list[n][m] = True
                        crash_list[m][n] = True
                        continue

    return [crash_list, crash_warning_list]


def comm_check(num_agents, uav, comm_range):

    group = np.zeros(num_agents, dtype=int)
    # initial group = [0, 0, 0, 0, 0, 0]
    # group = [1, 2, 2, 1, 3, 2] means agent_0 is part of group_1, agent_3 is part of group_1, ... 
    multi_hop_matrix  = np.zeros([num_agents, num_agents], dtype=bool) # Multi-hop comm matrix
    laplacian_matrix  = np.zeros([num_agents, num_agents], dtype=bool) # Laplacian matrix


    for n in range(num_agents):
        if group[n] == 0: # if this agent[n] did not assigned yet
            group[n] = group.max()+1
        for m in range(n+1, num_agents):
            uav_m = uav[m]
            uav_n = uav[n]
            m_n_dist = d_btw_points([uav_m.x, uav_m.y], [uav_n.x, uav_n.y])
            if m_n_dist < comm_range:
                if group[m] == 0: # if agent[m] did not assigned yet
                    group[m] = group[n]
                elif group[n] != group[m]:
                    # if agent[m] is already member of one group
                    # and the group is different from agent[n]
                    max_group_indx = max(group[n], group[m])
                    min_group_indx = min(group[n], group[m])
                    group[n] = group[m] = min_group_indx # change group number of agent[m] and agent[n] to lower
                    for o in range(m): # other assigned agents are checked too
                        if group[o] == max_group_indx:
                            group[o] = min_group_indx
    for g_n in set(group):
        # Multi-hop 
        multi_hop_matrix[group == g_n] = (group == g_n)
                
    for n in range(num_agents):
        laplacian_matrix[n][n] = True
        for m in range(n):
            uav_m = uav[m]
            uav_n = uav[n]
            m_n_dist = d_btw_points([uav_m.x, uav_m.y], [uav_n.x, uav_n.y])
            if m_n_dist < comm_range:
                laplacian_matrix[m][n] = True
                laplacian_matrix[n][m] = True

    return [group, multi_hop_matrix, laplacian_matrix]

def get_numpy_from_nonfixed_2d_array(origin_list, fixed_length, padding_value=0):
    rows = []
    for origin_row in origin_list:
        rows.append(np.pad(origin_row, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length)

def decentralized_obs_for_each_agent(global_obs, num_agents, laplacian_matrix):
    single_obs_size = int(len(global_obs)/num_agents)
    obs_matrix = np.zeros([num_agents, single_obs_size])
    for n in range(num_agents):
        obs_matrix[n,:] = global_obs[single_obs_size*(n):single_obs_size*(n+1)]
    comm_obs_list = []
    for n in range(num_agents):
        decentralized_comm_obs = []
        for m in chain(range(n, num_agents), range(n)):
            if laplacian_matrix[n][m]:
                decentralized_comm_obs.extend(obs_matrix[m,:].tolist() )
        comm_obs_list.append(decentralized_comm_obs)
    comm_obs = get_numpy_from_nonfixed_2d_array(comm_obs_list, len(global_obs))

#    for n in range(self.num_agents):
#        pad_size = len(global_obs) - comm_obs_list[n].size
#        comm_obs_list[n] = np.pad(comm_obs_list[n], (0,pad_size), 'constant', constant_values=-1)


    return comm_obs

def set_init_pos(np_random, court_lx, court_ly, gas, crash_range, num_agents, comm_range):
    ##------------------------------------- reset locations ----------------------------------------------
    init_pos = np.random.uniform(low=[5,5],high=[30,30], size=[num_agents,2])
    if num_agents > 1:
        while d_btw_points(init_pos[0], init_pos[1]) < crash_range:
            init_pos = np.random.uniform(low=[5,5],high=[30,30], size=[num_agents,2])
    return init_pos
    '''
    init_pos = np.ones([num_agents, 2]) # 2-Dim

    uni_dist = np_random.uniform

    cent_to_source = 0

    while cent_to_source < court_lx*0.3: # When c_to_s is bigger than court_lx*0.3, while is broken
        init_pos = np.array([[uni_dist(low =[5, 5], high=[court_lx-5, court_ly-5])]])
        cent_to_source = math.sqrt( (gas.S_x-init_pos[0,0])**2 + (gas.S_y-init_pos[0,1])**2 )

    for n in range(1,num_agents)
        init_pos_low  = [max(init_pos[n-1,0] - comm_range, 0),
                         max(init_pos[n-1,1] - comm_range, 0)]
        init_pos_high = [min(init_pos[n-1,0] + comm_range, court_lx),
                         min(init_pos[n-1,1] + comm_range, court_ly)]

        new_point = uni_dist(low =init_pos_low, high=init_pos_high)


    for n in range(num_agents):
        uav_to_source = 0
        if n==0:
            while uav_to_source < init_source_dist:
                init_pos[n,:] = uni_dist(low =init_pos_low, high=init_pos_high)
                uav_to_source = d_btw_points(init_pos[n], [gas.S_x, gas.S_y])
        else:
            for m in range(n):


                while (uav_to_source < init_source_dist or uav_to_uav < self.crash_range or uav_to_uav > self.wifi_range):
                    init_pos[n,:] = uni_dist(low =init_pos_low, high=init_pos_high)
                    uav_to_uav = d_btw_points(init_pos[n], init_pos[m])
                    uav_to_source = d_btw_points(init_pos[n], [gas.S_x, gas.S_y])
    return init_pos
    '''
