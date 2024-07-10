import math
import numpy as np
import gymnasium as gym

from .base import BaseEMTask


class FreeRecall(BaseEMTask):
    def __init__(self, vocabulary_num=20, memory_num=5, memory_var=0, retrieve_time_limit=None, true_reward=1.0, false_reward=-0.1, repeat_penalty=-0.1, 
    not_know_reward=-0.1, reset_state_before_test=False, start_recall_cue=False, encode_reward_weight=0.0, return_action=False, return_reward=False, 
    #forward_smooth=0, backward_smooth=0, 
    seed=None,
    dt=10, tau=10):
        super().__init__(reset_state_before_test=reset_state_before_test, seed=seed)
        self.vocabulary_num = vocabulary_num        # dimension of items
        self.memory_num = memory_num                # sequence length
        self.memory_var = memory_var                # variance of memory sequence length
        assert memory_num > memory_var
        self.current_memory_num = memory_num        # current memory sequence length
        # rewards and penalties
        self.true_reward = true_reward
        self.false_reward = false_reward
        self.not_know_reward = not_know_reward
        self.repeat_penalty = repeat_penalty

        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else memory_num
        self.current_retrieve_time_limit = max(self.retrieve_time_limit, self.current_memory_num)
        self.start_recall_cue = start_recall_cue                            # add a cue at the beginning of recall
        self.encode_reward_weight = encode_reward_weight                    # weight of reward during encoding
        self.return_action = return_action                                  # return last action as part of observation
        self.return_reward = return_reward                                  # return last reward as part of observation

        self.steps_each_item = int(tau / dt)            # for CTRNN, show multiple steps for each item
        self.batch_size = 1

        # self.forward_smooth = forward_smooth            # add weighted last item to current item
        # self.backward_smooth = backward_smooth          # add weighted next item to current item
        # assert self.forward_smooth >= 0 and self.forward_smooth <= 1
        # assert self.backward_smooth >= 0 and self.backward_smooth <= 1
        # self.smooth_matrix = self.generate_smooth_matrix()  # generate smooth matrix

        self.memory_sequence = self.generate_sequence()     # generate memory sequence
        self.stimuli = self.generate_stimuli()              # generate stimuli according to memory sequence
        self.current_timestep = 0                           # reset current timestep
        self.current_step_within_item = 0                   # for CTRNN, when there's multiple steps for each item
        self.testing = False                                # false: encoding phase, true: recall phase
        self.not_retrieved = np.ones((self.vocabulary_num+1), dtype=bool)   # flag for each item, 0 for retrieved, 1 for not retrieved
        self.reported_memory = 0

        obs_shape = self.vocabulary_num+1
        if self.start_recall_cue:
            obs_shape += 1
        if self.return_action:
            obs_shape += self.vocabulary_num+1
        if self.return_reward:
            obs_shape += 1
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_shape,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.vocabulary_num+1)

    def reset(self):
        """
        reset the trial, return the first observation
        """
        self.current_memory_num = np.random.randint(self.memory_num - self.memory_var, self.memory_num + self.memory_var + 1)
        self.memory_sequence = self.generate_sequence()
        self.stimuli = self.generate_stimuli()
        self.current_retrieve_time_limit = max(self.retrieve_time_limit, self.current_memory_num)

        self.current_timestep = 0
        self.current_step_within_item = 0
        self.testing = False
        self.not_retrieved = np.ones((self.vocabulary_num+1), dtype=bool)
        self.reported_memory = 0
        info = {"phase": "encoding"}
        observations = self.stimuli[self.current_timestep, :]
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.zeros(1)), axis=0)
        if self.return_action:
            observations = np.concatenate((observations, np.zeros((self.vocabulary_num+1))), axis=0)
        if self.return_reward:
            observations = np.concatenate((observations, np.zeros(1)), axis=0)
        # observations = observations.reshape(1, -1)

        # print(self.memory_sequence)
        return observations, info

    def step(self, action):
        """
        for reinforcement learning
        """
        # actions = np.array([action.cpu().detach().numpy() for action in actions])
        # action = action.cpu().detach().numpy()

        start_recall = 0
        # compute returned action, if needed
        if self.return_action:
            returned_action = self.get_returned_action(action)
        # recall phase, compute rewards
        if self.testing:
            reward = self.compute_reward(action)
            observations = np.zeros((self.vocabulary_num+1))
            info = {"phase": "recall", "gt": 0, "gt_mask": False}
            self.increase_timestep()
            done = self.check_done()
        else:
            reward = 0.0
            done = False
            if self.encode_reward_weight > 0:
                # during encoding, train the model to output the just inputed item
                eq = action == self.memory_sequence[self.current_timestep]
                    # np.equal(actions, self.memory_sequence[:, self.current_timestep]).astype(int)
                reward = eq * self.true_reward * self.encode_reward_weight + (1 - eq) * self.false_reward * self.encode_reward_weight
                # if action == self.memory_sequence[self.current_timestep]:
                #     rewards = self.true_reward * self.encode_reward_weight
                # else:
                #     rewards = self.false_reward * self.encode_reward_weight
            self.increase_timestep()
            if self.current_timestep == self.current_memory_num:
                # before the first timestep of the recall phase
                self.testing = True
                observations = np.zeros((self.vocabulary_num+1))
                self.increase_timestep(set_zero=True)
                info = {"phase": "recall", "gt": self.memory_sequence[self.current_timestep-1], "gt_mask": True}
                if self.reset_state_before_test:    # send signal for the agent to reset its state
                    info["reset_state"] = True
                start_recall = 1
            else:
                observations = self.stimuli[self.current_timestep, :]
                info = {"phase": "encoding", "gt": self.memory_sequence[self.current_timestep-1], "gt_mask": True}
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.ones(1) * start_recall), axis=0)
        if self.return_action:
            if self.testing and self.current_timestep > 0:
                observations = np.concatenate((observations, returned_action), axis=0)
            else:
                observations = np.concatenate((observations, np.zeros((self.vocabulary_num+1))), axis=0)
        if self.return_reward:
            observations = np.concatenate((observations, np.array([reward])), axis=0)
        # observations = observations.reshape(1, -1)
        return observations, reward, done, False, info

    # def generate_smooth_matrix(self):
    #     """
    #     compute smooth matrix
        
    #     when generating stimuli, multiply smooth matrix with one-hot encoding of memory sequence

    #     dim of smooth matrix: memory_num * memory_num
    #     """
    #     smooth_matrix = np.eye(self.current_memory_num)
    #     for i in range(self.current_memory_num - 1):
    #         for j in range(self.current_memory_num - i - 1):
    #             smooth_matrix[j+i+1][j] = math.pow(self.forward_smooth, i+1)
    #             smooth_matrix[j][j+i+1] = math.pow(self.backward_smooth, i+1)
    #     return smooth_matrix

    def generate_sequence(self):
        """
        generate memory sequence, each item is a number from 1 to vocabulary_num

        output: batch_size * memory_num
        """
        memory_sequence = np.zeros((self.current_memory_num), dtype=int)
        memory_sequence = np.random.choice(self.vocabulary_num, self.current_memory_num, replace=False) + 1
        return memory_sequence      # axis 0: batch size, axis 1: length of sequence

    def generate_stimuli(self):
        """
        generate stimuli according to memory sequence

        when there's no smoothing, turn memory sequence into one-hot encoding

        output: memory_num * (vocabulary_num+1)
        """
        data = np.eye(self.vocabulary_num+1)[self.memory_sequence]
        # self.smooth_matrix = self.generate_smooth_matrix()
        # data = (self.smooth_matrix @ data).T
        # data = data / np.linalg.norm(data, axis=0)
        # data = data.T
        return data

    def increase_timestep(self, set_zero=False):
        """
        decide whether to increase timestep

        for CTRNN, first add to current_step_within_item, until it reaches steps_each_item, then increase current_timestep
        when set_zero is True, reset current_timestep to 0
        """
        self.current_step_within_item += 1
        if self.current_step_within_item == self.steps_each_item:
            self.current_step_within_item = 0
            if set_zero:
                self.current_timestep = 0
            else:
                self.current_timestep += 1

    def get_returned_action(self, action):
        """
        compute action to return

        for the first timestep, return all zeros
        for the following timesteps, return one-hot encoding of action

        input: action, int
        output: returned_actions, shape=vocabulary_num+1
        """

        if self.current_timestep == 0 and not self.testing:
            returned_action = np.zeros((self.vocabulary_num+1))
        else:
            returned_action = np.eye(self.vocabulary_num+1)[action].reshape(-1)
        return returned_action
    
    def compute_reward(self, action):
        """
        compute rewards according to actions

        if action is in memory_sequence and hasn't been retrieved, give true reward
        if action is in memory_sequence but has been retrieved, give repeat penalty
        if action is not in memory_sequence, give false reward
        if action is 0, give not_know_reward

        input: action, int
        output: rewards, float
        """
        if action in list(self.memory_sequence):
            if self.not_retrieved[action]:
                reward = self.true_reward
                self.not_retrieved[action] = False
            elif np.sum(self.not_retrieved) == 0:
                reward = 0.0
            else:
                reward = self.repeat_penalty
            self.reported_memory += 1
        elif action == 0:
            reward = self.not_know_reward
        else:
            reward = self.false_reward
            self.reported_memory += 1
        return reward
    
    def check_done(self):
        """
        check whether the trial has completed
        """
        if self.current_timestep >= self.current_retrieve_time_limit:
            return True
        else:
            return self.reported_memory >= self.current_memory_num and np.sum(self.not_retrieved) == 0
                # np.logical_and(self.reported_memory >= self.current_memory_num, np.sum(self.not_retrieved, axis=1) == 0)
        # if self.current_timestep >= self.current_retrieve_time_limit or (self.reported_memory >= self.current_memory_num).all() \
        #     or (np.sum(self.not_retrieved, axis=1) == 0).all():
        #     return True
        # else:
        #     return False

    def render(self, mode='human'):
        pass

    # def get_batch(self):
    #     """
    #     for supervised learning, return inputs and ground truth data for all timesteps
    #     """
    #     data = np.zeros((self.current_memory_num + self.current_retrieve_time_limit, self.vocabulary_num+1))
    #     gt = np.zeros((self.current_memory_num + self.current_retrieve_time_limit, self.vocabulary_num+1))
    #     data[:self.current_memory_num, :] = self.stimuli.transpose(1, 0, 2)
    #     gt[self.current_memory_num:self.current_memory_num*2, :] = np.eye(self.vocabulary_num+1)[self.memory_sequence]
    #     if self.steps_each_item > 1:
    #         data = np.repeat(data, self.steps_each_item, axis=0)
    #         gt = np.repeat(gt, self.steps_each_item, axis=0)
    #     if self.start_recall_cue:
    #         cue = np.zeros((self.current_memory_num + self.current_retrieve_time_limit, 1))
    #         cue[self.current_memory_num, 0] = 1
    #         data = np.concatenate((data, cue), axis=1)
    #     return data, gt

    def compute_accuracy(self, actions):
        """
        compute accuracy for all timesteps

        input: actions, (timesteps, batch_size)
        outputs: number of three types of results: correct, wrong, not_know, i.e. 3 int
        """
        # actions = np.array(actions).T
        # print(actions[self.current_memory_num:], self.memory_sequence)

        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0

        not_retrieved = np.ones((self.vocabulary_num+1), dtype=bool)
        
        # for action_batch in actions[self.current_memory_num:]:
        #     for i, action in enumerate(action_batch):
        #         action = int(action)
        #         if action in self.memory_sequence[i] and not_retrieved[i, action]:
        #             correct_actions += 1
        #             not_retrieved[i, action] = False
        #         elif action == 0:
        #             not_know_actions += 1
        #         else:
        #             wrong_actions += 1

        for action in actions[self.current_memory_num:]:
            action = int(action)
            if action in self.memory_sequence and not_retrieved[action]:
                correct_actions += 1
                not_retrieved[action] = False
            elif action == 0:
                not_know_actions += 1
            else:
                wrong_actions += 1

        return correct_actions, wrong_actions, not_know_actions

    # def compute_rewards(self, actions):
    #     """
    #     compute rewards for all timesteps
        
    #     input: actions, timesteps
    #     output: rewards, timesteps
    #     """
    #     rewards = []
    #     not_retrieved = np.ones(self.vocabulary_num+1, dtype=bool)
    #     for t, action in enumerate(actions):
    #             if t < self.current_memory_num:
    #                 rewards.append(0.0)
    #             else:
    #                 if action in list(self.memory_sequence):
    #                     if not_retrieved[action]:
    #                         rewards.append(self.true_reward)
    #                         not_retrieved[action] = False
    #                     else:
    #                         rewards.append(self.repeat_penalty)
    #                 elif action == 0:
    #                     rewards.append(self.not_know_reward)
    #                 else:
    #                     rewards.append(self.false_reward)
    #     return np.array(rewards).transpose(1, 0)
    
    def get_ground_truth(self, phase="all"):
        """
        get ground truth data for all timesteps

        output: ground truth with numbers (not one-hot), memory_num
        """
        # gt = np.concatenate((self.memory_sequence, self.memory_sequence), axis=1)
        gt = np.concatenate((self.memory_sequence, self.memory_sequence), axis=0)

        if phase == 'encoding':
            mask = np.concatenate((np.ones(self.memory_num), np.zeros(self.memory_num))).astype(int)
        elif phase == 'recall':
            mask = np.concatenate((np.zeros(self.memory_num), np.ones(self.memory_num))).astype(int)
        else:
            mask = np.ones(self.memory_num*2).astype(int)

        return gt, mask
    
    def get_trial_data(self):
        """
        get trial data
        """
        return self.memory_sequence


# test
if __name__ == "__main__":
    env = FreeRecall(vocabulary_num=10, memory_num=5, retrieve_time_limit=5)


