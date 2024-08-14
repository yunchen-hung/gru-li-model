import math
import numpy as np
import gymnasium as gym

from .base import BaseEMTask


class FreeRecallRepeat(BaseEMTask):
    def __init__(self, vocabulary_num=20, memory_num=5, memory_var=0, retrieve_time_limit=None, true_reward=1.0, false_reward=-0.1, repeat_penalty=-0.1, 
    not_know_reward=-0.1, reset_state_before_test=False, start_recall_cue=False, encode_reward_weight=0.0, return_action=False, return_reward=False, 
    dt=10, tau=10, repeat_times=(1,3)):
        super().__init__(reset_state_before_test=reset_state_before_test)
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
        self.repeat_times = np.arange(repeat_times[0], repeat_times[1]+1)

        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else memory_num
        self.current_retrieve_time_limit = max(self.retrieve_time_limit, self.current_memory_num)
        self.start_recall_cue = start_recall_cue                            # add a cue at the beginning of recall
        self.encode_reward_weight = encode_reward_weight                    # weight of reward during encoding
        self.return_action = return_action                                  # return last action as part of observation
        self.return_reward = return_reward                                  # return last reward as part of observation

        self.steps_each_item = int(tau / dt)            # for CTRNN, show multiple steps for each item
        self.batch_size = 1

        self.current_memory_num = np.random.randint(self.memory_num - self.memory_var, self.memory_num + self.memory_var + 1)
        self.current_repeat_times = np.random.choice(self.repeat_times)
        self.memory_sequence, self.not_recalled_basket = self.generate_sequence()     # generate memory sequence
        self.stimuli = self.generate_stimuli()              # generate stimuli according to memory sequence
        self.current_timestep = 0                           # reset current timestep
        self.current_step_within_item = 0                   # for CTRNN, when there's multiple steps for each item
        self.testing = False                                # false: encoding phase, true: recall phase
        # self.not_retrieved = np.ones((self.vocabulary_num+1), dtype=bool)   # flag for each item, 0 for retrieved, 1 for not retrieved
        self.not_recalled_basket = np.zeros((self.vocabulary_num+1), dtype=int)  # basket for not recalled items (>0 means the number of not recalled items)
        self.reported_memory = 0

    def reset(self, batch_size=1, regenerate_contexts=True):
        """
        reset the trial, return the first observation
        """
        if regenerate_contexts:
            self.current_memory_num = np.random.randint(self.memory_num - self.memory_var, self.memory_num + self.memory_var + 1)
            self.current_repeat_times = np.random.choice(self.repeat_times)
            self.memory_sequence, self.not_recalled_basket = self.generate_sequence(batch_size=batch_size)
            self.stimuli = self.generate_stimuli(batch_size=batch_size)
            self.current_retrieve_time_limit = max(self.retrieve_time_limit, self.current_memory_num)

        self.current_timestep = 0
        self.current_step_within_item = 0
        self.testing = False
        # self.not_retrieved = np.ones((batch_size, self.vocabulary_num+1), dtype=bool)
        self.reported_memory = np.zeros(batch_size)
        info = {"phase": "encoding"}
        observations = self.stimuli[:, self.current_timestep, :]
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.zeros((batch_size, 1))), axis=1)
        if self.return_action:
            observations = np.concatenate((observations, np.zeros((batch_size, self.vocabulary_num+1))), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, np.zeros((batch_size, 1))), axis=1)
        return observations, info

    def step(self, actions):
        """
        for reinforcement learning
        """
        batch_size = len(actions)
        actions = np.array([action.cpu().detach().numpy() for action in actions])

        start_recall = 0
        # compute returned action, if needed
        if self.return_action:
            returned_action = self.get_returned_action(actions)
        # recall phase, compute rewards
        if self.testing:
            rewards = self.compute_reward(actions)
            observations = np.zeros((batch_size, self.vocabulary_num+1))
            info = {"phase": "recall"}
            self.increase_timestep()
            done = self.check_done()
        else:
            rewards = np.zeros(batch_size)
            done = np.zeros(batch_size, dtype=bool)
            if self.encode_reward_weight > 0:
                # during encoding, train the model to output the just inputed item
                eq = np.equal(actions, self.memory_sequence[:, self.current_timestep]).astype(int)
                rewards = eq * self.true_reward * self.encode_reward_weight + (1 - eq) * self.false_reward * self.encode_reward_weight
            self.increase_timestep()
            if self.current_timestep == self.current_memory_num:
                # before the first timestep of the recall phase
                self.testing = True
                observations = np.zeros((batch_size, self.vocabulary_num+1))
                self.increase_timestep(set_zero=True)
                info = {"phase": "recall"}
                if self.reset_state_before_test:    # send signal for the agent to reset its state
                    info["reset_state"] = True
                start_recall = 1
            else:
                observations = self.stimuli[:, self.current_timestep, :]
                info = {"phase": "encoding"}
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.ones((batch_size, 1)) * start_recall), axis=1)
        if self.return_action:
            if self.testing and self.current_timestep > 0:
                observations = np.concatenate((observations, returned_action), axis=1)
            else:
                observations = np.concatenate((observations, np.zeros((batch_size, self.vocabulary_num+1))), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, rewards.reshape(-1, 1)), axis=1)
        # observations = observations.reshape(1, -1)
        return observations, rewards, done, info

    def generate_sequence(self, batch_size=1):
        """
        generate memory sequence, each item is a number from 1 to vocabulary_num

        output: batch_size * memory_num
        """
        memory_sequence = np.zeros((batch_size, self.current_memory_num), dtype=int)
        basket = np.zeros((batch_size, self.vocabulary_num+1), dtype=int)
        for i in range(batch_size):
            items = np.random.choice(self.vocabulary_num, self.current_memory_num-self.current_repeat_times, replace=False) + 1
            repeated_items = np.random.choice(items, self.current_repeat_times, replace=True)
            memory_sequence[i] = np.random.permutation(np.concatenate((items, repeated_items)))
            basket[i, items] += 1
            for j in repeated_items:
                basket[i, j] += 1
        return memory_sequence, basket      # memory_sequence: batch size * length of sequence

    def generate_stimuli(self, batch_size=1):
        """
        generate stimuli according to memory sequence

        when there's no smoothing, turn memory sequence into one-hot encoding

        output: memory_num * (vocabulary_num+1)
        """
        data = np.eye(self.vocabulary_num+1)[self.memory_sequence]
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

    def get_returned_action(self, actions):
        """
        compute action to return

        for the first timestep, return all zeros
        for the following timesteps, return one-hot encoding of action

        input: action, int
        output: returned_actions, shape=vocabulary_num+1
        """
        batch_size = len(actions)

        if self.current_timestep == 0 and not self.testing:
            returned_action = np.zeros((batch_size, self.vocabulary_num+1))
        else:
            returned_action = np.eye(self.vocabulary_num+1)[actions]
        return returned_action
    
    def compute_reward(self, actions):
        """
        compute rewards according to actions

        if action is in memory_sequence and hasn't been retrieved, give true reward
        if action is in memory_sequence but has been retrieved, give repeat penalty
        if action is not in memory_sequence, give false reward
        if action is 0, give not_know_reward

        input: action, list of int, (batch_size)
        output: rewards, list of float, (batch_size)
        """
        rewards = []
        for i, action in enumerate(actions):
            if action in list(self.memory_sequence[i]):
                if self.not_recalled_basket[i][action] > 0:
                    rewards.append(self.true_reward)
                    self.not_recalled_basket[i][action] -= 1
                    # self.not_retrieved[i][action] = False
                elif np.sum(self.not_recalled_basket[i]) == 0:
                # elif np.sum(self.not_retrieved[i]) == 0:
                    rewards.append(0.0)
                else:
                    rewards.append(self.repeat_penalty)
                self.reported_memory[i] += 1
            elif action == 0:
                rewards.append(self.not_know_reward)
            else:
                rewards.append(self.false_reward)
                self.reported_memory[i] += 1
        return np.array(rewards)
    
    def check_done(self):
        """
        check whether the trial has completed
        """
        if self.current_timestep >= self.current_retrieve_time_limit:
            return np.ones(len(self.reported_memory), dtype=bool)
        else:
            return np.logical_and(self.reported_memory >= self.current_memory_num, np.sum(self.not_recalled_basket, axis=1) == 0)

    def render(self, mode='human'):
        pass

    def compute_accuracy(self, actions):
        """
        compute accuracy for all timesteps

        input: actions, (timesteps, batch_size)
        outputs: number of three types of results: correct, wrong, not_know, i.e. 3 int
        """
        batch_size = len(actions)

        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0

        not_recalled_basket = np.zeros((batch_size, self.vocabulary_num+1), dtype=int)
        for i in range(self.memory_sequence.shape[0]):
            for j in range(self.memory_sequence.shape[1]):
                not_recalled_basket[i, self.memory_sequence[i, j]] += 1

        # not_retrieved = np.ones((batch_size, self.vocabulary_num+1), dtype=bool)
        
        for action_batch in actions[self.current_memory_num:]:
            for i, action in enumerate(action_batch):
                action = int(action)
                # if action in self.memory_sequence[i] and not_retrieved[i, action]:
                if action in self.memory_sequence[i] and not_recalled_basket[i, action] > 0:
                    correct_actions += 1
                    # not_retrieved[i, action] = False
                    not_recalled_basket[i, action] -= 1
                elif action == 0:
                    not_know_actions += 1
                else:
                    wrong_actions += 1
        return correct_actions, wrong_actions, not_know_actions
    
    def get_ground_truth(self, phase="all"):
        """
        get ground truth data for all timesteps

        output: ground truth with numbers (not one-hot), memory_num
        """
        gt = np.concatenate((self.memory_sequence, self.memory_sequence), axis=1)

        if phase == 'encoding':
            mask = np.concatenate((np.ones(self.memory_num), np.zeros(self.memory_num))).astype(int)
        elif phase == 'recall':
            mask = np.concatenate((np.zeros(self.memory_num), np.ones(self.memory_num))).astype(int)
        else:
            mask = np.ones(self.memory_num*2).astype(int)

        return gt, mask.reshape(1, -1)
    
    def get_trial_data(self):
        """
        get trial data
        """
        return self.memory_sequence


# test
if __name__ == "__main__":
    env = FreeRecallRepeat(vocabulary_num=10, memory_num=5, retrieve_time_limit=5)


