import gymnasium as gym 
from gymnasium import Wrapper, spaces
import numpy as np


class MetaLearningEnv(Wrapper):
    metadata = {'render_modes': ['human','rgb_array']}
    def __init__(self,env):
        super().__init__(env)
        self.env=env 
        self.prev_action=self.env.action_space.sample()
        if hasattr(self.env,'convert_action_to_observation'):
            self.prev_action=self.env.convert_action_to_observation(self.prev_action)
        self.prev_reward=0

    def step(self,action):
        obs, reward, terminated, info = self.env.step(action)
        if hasattr(self.env, 'convert_action_to_observation'):
            action = self.env.convert_action_to_observation(action)
        self.prev_action = action
        self.prev_reward = reward
        obs_wrapped = np.hstack([obs, self.prev_action, self.prev_reward])
        return obs_wrapped,reward,terminated,info
    
    def reset(self):
        obs,info=self.env.reset()
        self.prev_action = self.env.action_space.sample()
        if hasattr(self.env, 'convert_action_to_observation'):
            self.prev_action = self.env.convert_action_to_observation(self.prev_action)
        self.prev_reward=0
        obs_wrapped=np.hstack([obs,self.prev_action,self.prev_reward])
        return obs_wrapped,info 

