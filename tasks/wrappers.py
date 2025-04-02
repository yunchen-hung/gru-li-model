import gymnasium as gym 
from gymnasium import Wrapper, spaces
import numpy as np


class MetaLearningEnv(Wrapper):
    metadata = {'render_modes': ['human','rgb_array']}
    def __init__(self,env):
        super().__init__(env)
        self.env=env 
        self.prev_action=self.env.action_space.sample()
        self.prev_action = self.env._convert_action_to_observation(self.prev_action)
        # if hasattr(self.env,'convert_action_to_observation'):
        #     self.prev_action=self.env.unwrapped.convert_action_to_observation(self.prev_action)
        self.prev_reward=0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
            shape=(self.env.observation_space.shape[0]+self.env.action_shape+1,), dtype=np.float32)
        # print(self.env.observation_space.shape[0], self.env.action_space.n)
        # self.action_space = spaces.Discrete(self.env.action_space.n)
        self.action_space = env.action_space

    def step(self,action):
        obs, reward, done, _, info = self.env.step(action)
        # if hasattr(self.env, 'convert_action_to_observation'):
        #     action = self.env.unwrapped.convert_action_to_observation(action)
        action = self.env._convert_action_to_observation(action)
        obs_wrapped = np.hstack([obs.reshape(-1), action, reward])
        self.prev_action = action
        self.prev_reward = reward
        return obs_wrapped,reward,done,_,info
    
    def reset(self, **kwargs):
        obs,info=self.env.reset(**kwargs)
        # self.prev_action = self.env.action_space.sample()
        # if hasattr(self.env, 'convert_action_to_observation'):
        #     self.prev_action = self.env.convert_action_to_observation(self.prev_action)
        # self.prev_reward = 0
        obs_wrapped = np.hstack([obs.reshape(-1),self.prev_action,self.prev_reward])
        # self.prev_reward = 0
        # print('meta', obs.shape, obs_wrapped.shape)
        return obs_wrapped,info

    def render(self, mode='human'):
        pass
    

class PlaceHolderWrapper(Wrapper):
    metadata = {'render_modes': ['human','rgb_array']}
    def __init__(self,env, placeholder_dim):
        super().__init__(env)
        self.env=env
        self.placeholder_dim = placeholder_dim
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
            shape=(self.env.observation_space.shape[0]+self.placeholder_dim,), dtype=np.float32)
        # print(self.observation_space.shape[0])
        self.action_space = spaces.Discrete(self.env.action_space.n)
    
    def step(self,action):
        obs, reward, done, terminated, info = self.env.step(action)
        obs_wrapped = np.hstack([obs.reshape(-1), np.zeros(self.placeholder_dim)])
        return obs_wrapped,reward, done,terminated,info
    
    def reset(self, batch_size=1):
        obs,info=self.env.reset()
        obs_wrapped = np.hstack([obs.reshape(-1), np.zeros(self.placeholder_dim)])
        # print('place', obs.shape, obs_wrapped.shape)
        return obs_wrapped,info
