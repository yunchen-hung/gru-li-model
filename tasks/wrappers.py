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
        try:
            action = action[0].item()
        except:
            pass
        if hasattr(self.env, 'convert_action_to_observation'):
            action = self.env.convert_action_to_observation(action)
        obs_wrapped = np.hstack([obs.reshape(-1), action, self.prev_reward]).reshape(1, -1)
        self.prev_action = action
        self.prev_reward = reward[0]
        return obs_wrapped,reward,terminated,info
    
    def reset(self, batch_size=1):
        obs,info=self.env.reset()
        # self.prev_action = self.env.action_space.sample()
        # if hasattr(self.env, 'convert_action_to_observation'):
        #     self.prev_action = self.env.convert_action_to_observation(self.prev_action)
        # self.prev_reward = 0
        obs_wrapped = np.hstack([obs.reshape(-1),self.prev_action,self.prev_reward]).reshape(1, -1)
        self.prev_reward = 0
        return obs_wrapped,info 


class OriginMetaLearningEnv(Wrapper):
    """
    return action as one-hot vectors instead of converting it to stimuli format
    """
    metadata = {'render_modes': ['human','rgb_array']}
    def __init__(self,env):
        super().__init__(env)
        self.env=env 
        self.prev_action=self.env.action_space.sample()
        # convert action to one-hot vector
        self.prev_action = np.eye(self.env.action_space.n)[self.prev_action]
        self.prev_reward=0

    def step(self,action):
        obs, reward, terminated, info = self.env.step(action)
        try:
            action = action[0].item()
        except:
            pass
        action = np.eye(self.env.action_space.n)[action]
        obs_wrapped = np.hstack([obs.reshape(-1), self.prev_action, self.prev_reward]).reshape(1, -1)
        self.prev_action = action
        self.prev_reward = reward[0]
        return obs_wrapped,reward,terminated,info
    
    def reset(self):
        obs,info=self.env.reset()
        self.prev_action = self.env.action_space.sample()
        self.prev_action = np.eye(self.env.action_space.n)[self.prev_action]
        self.prev_reward=0
        obs_wrapped=np.hstack([obs.reshape(-1),self.prev_action,self.prev_reward]).reshape(1, -1)
        return obs_wrapped,info
