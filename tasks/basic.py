import math
import numpy as np
import gym

# TODO
class BasicTask(gym.Env):
    def __init__(self, batch_size):
        self.batch_size = batch_size
