import numpy as np
import gym


class FreeRecall(gym.Env):
    def __init__(self, vocabulary_num=100, memory_num=10, retrieve_time_limit=15, true_reward=1.0, false_reward=-0.1, reset_state_before_test=False):
        self.vocabulary_num = vocabulary_num
        self.memory_num = memory_num
        self.true_reward = true_reward
        self.false_reward = false_reward
        self.retrieve_time_limit = max(retrieve_time_limit, memory_num)
        self.reset_state_before_test = reset_state_before_test
        self.batch_size = 1

        self.memory_sequence = self.generate_sequence()
        self.current_timestep = 0
        self.testing = False    # false: presenting sequence, true: testing
        self.not_retrieved = np.zeros(self.vocabulary_num+1, dtype=np.bool)
        self.not_retrieved[self.memory_sequence] = True
        self.reported_memory = 0

    def generate_sequence(self):
        rand_index = np.arange(1, self.vocabulary_num+1)
        np.random.shuffle(rand_index)
        return rand_index[:self.memory_num]
    
    def step(self, action):
        if self.testing:
            if action in list(self.memory_sequence) and self.not_retrieved[action]:
                reward = self.true_reward
                self.not_retrieved[action] = False
                self.reported_memory += 1
            elif action == 0:
                reward = 0.0    # not know
            else:
                reward = self.false_reward
                self.reported_memory += 1
            observation = np.zeros(self.vocabulary_num+1)
            info = {"encoding_on": False}
            self.current_timestep += 1
            if self.current_timestep >= self.retrieve_time_limit or self.reported_memory >= self.memory_num or np.sum(self.not_retrieved) == 0:
                done = True
            else:
                done = False
        else:
            reward = 0.0
            done = False
            self.current_timestep += 1
            if self.current_timestep == self.memory_num:
                self.testing = True
                observation = np.zeros(self.vocabulary_num+1)
                self.current_timestep = 0
                info = {"encoding_on": False}
                if self.reset_state_before_test:
                    info["reset_state"] = True
            else:
                observation = np.eye(self.vocabulary_num+1)[self.memory_sequence[self.current_timestep]]
                info = {"encoding_on": True, "retrieval_off": True}
        return observation, reward, done, info

    def reset(self):
        self.memory_sequence = self.generate_sequence()
        self.current_timestep = 0
        self.testing = False
        self.not_retrieved = np.zeros(self.vocabulary_num+1, dtype=np.bool)
        self.not_retrieved[self.memory_sequence] = True
        self.reported_memory = 0
        info = {"encoding_on": True, "retrieval_off": True}
        return np.eye(self.vocabulary_num+1)[self.memory_sequence[self.current_timestep]], info

    def render(self, mode='human'):
        pass

    def compute_accuracy(self, actions):
        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0
        not_retrieved = np.zeros(self.vocabulary_num+1, dtype=np.bool)
        not_retrieved[self.memory_sequence] = True
        for action in actions[self.memory_num:]:
            if action in list(self.memory_sequence) and not_retrieved[action]:
                correct_actions += 1
                not_retrieved[action] = False
            elif action == 0:
                not_know_actions += 1
            else:
                wrong_actions += 1
        return correct_actions, wrong_actions, not_know_actions
