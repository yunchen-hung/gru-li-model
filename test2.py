import numpy as np
import random
import gymnasium as gym
# import torch
# from environment import *
# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


class HarlowEnv(gym.Env):
    """
    A bandit environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, flip_prob = 0.2):
        """
        Construct an environment.
        """

        # max number of trials per episode
        self.num_trials = 20

        # flip probability
        self.flip_prob = flip_prob

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (1,))


    def reset(self, seed = None, option = {}):
        """
        Reset the environment.
        """

        np.random.seed(seed)
        random.seed(seed)

        # reset the environment
        self.num_completed = 0
        self.stage = 'fixation'
        self.correct_answer = np.random.randint(0, 2)

        obs = np.array([1.])
        info = {
            'correct_answer': self.correct_answer,
        }

        return obs, info
    

    def step(self, action):
        """
        Step the environment.
        """

        done = False

        # fixation stage
        if self.stage == 'fixation':
            self.stage = 'decision'

            # fixation action
            if action == 2:
                reward = 0.
            
            # decision action
            else:
                reward = -1.
            
            obs = np.array([0.])
        
        # decision stage
        elif self.stage == 'decision':
            self.stage = 'fixation'
            self.num_completed += 1
            self.flip_bandit()

            if action == self.correct_answer:
                reward = 1.
            else:
                reward = -1.
            
            obs = np.array([1.])
        
        if self.num_completed >= self.num_trials + np.random.randint(0, 10, 1): # test for different episode length
            done = True
        
        info = {
            'correct_answer': self.correct_answer,
        }

        return obs, reward, done, False, info
    

    def flip_bandit(self):
        """
        Flip the bandit.
        """

        if np.random.random() < self.flip_prob:
            self.correct_answer = 1 - self.correct_answer

    
    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot


if __name__ == '__main__':
    batch_size = 3
    env = gym.vector.AsyncVectorEnv([
        lambda: HarlowEnv()
        for _ in range(batch_size)
    ])
    seeds = np.random.randint(0, 1000, batch_size)
    obs, info = env.reset(seed=seeds)
    print('initial obs:', obs.reshape(-1))
    dones = np.zeros(batch_size, dtype = bool)
    while not all(dones):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(
            'obs:', obs.reshape(-1), '|',
            'action:', action, '|',
            'correct answer:', info['correct_answer'], '|',
            'reward:', reward, '|',
            'done:', done, '|',
        )
        dones = np.logical_or(dones, done)