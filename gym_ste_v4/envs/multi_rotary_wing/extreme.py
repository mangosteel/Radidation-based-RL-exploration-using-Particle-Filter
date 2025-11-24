import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste_v4.envs.multi_rotary_wing.base import *
from gym.envs.registration import register

from datetime import datetime


class MultiRotaryExtEnv(BaseEnv):  
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        BaseEnv.__init__(self)

        #self.reset()

    def _set_init_state(self):
        # set initial state randomly
        self.radiation.S_x = self.np_random.uniform(low=self.court_lx*0.2, high=self.court_lx*0.8) # 가스대신에 방사선으로 !
        self.radiation.S_y = self.np_random.uniform(low=self.court_ly*0.2, high=self.court_ly*0.8)

        self.radiation.d = self.np_random.uniform(low=5, high=15)                # diffusivity [10m^2/s]
        self.radiation.t = self.np_random.uniform(low=500, high=1500)            # radiation life time [1000se$
        self.radiation.radi_x = self.np_random.uniform(low=1000, high=3000)           # radiation strength >> activity로 수정해야댐!

        self.t_wind.mean_phi = self.np_random.uniform(low=0, high=360)        # mean wind direction
        self.t_wind.mean_speed = self.np_random.uniform(low=0, high=10)       # 바람정보 안씀!

register( # 개별적인 환경설정 등록 / multi-rotary-env의 init 모듈에서 extreme을 호출한다!
    id='MultiRotaryExtEnv-v0',
    entry_point='gym_ste_v4.envs.multi_rotary_wing:MultiRotaryExtEnv',
)

