import math
import numpy as np
import gym


class FreeRecall(gym.Env):
    def __init__(self, vocabulary_num=20, memory_num=5, memory_var=0, retrieve_time_limit=5, true_reward=1.0, false_reward=-0.1, repeat_penalty=-0.1, 
    not_know_reward=-0.1, reset_state_before_test=False, start_recall_cue=False, encode_reward_weight=0.0, return_action=False, return_reward=False, 
    forward_smooth=0, backward_smooth=0, dt=10, tau=10, batch_size=1):
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

        self.retrieve_time_limit = max(retrieve_time_limit, memory_num)     # maximum time steps for retrieval, must > memory_num
        self.reset_state_before_test = reset_state_before_test              # reset state of the model before testing
        self.start_recall_cue = start_recall_cue                            # add a cue at the beginning of recall
        self.batch_size = batch_size
        self.encode_reward_weight = encode_reward_weight                    # weight of reward during encoding
        self.return_action = return_action                                  # return last action as part of observation
        self.return_reward = return_reward                                  # return last reward as part of observation

        self.steps_each_item = int(tau / dt)            # for CTRNN, show multiple steps for each item

        self.forward_smooth = forward_smooth            # add weighted last item to current item
        self.backward_smooth = backward_smooth          # add weighted next item to current item
        assert self.forward_smooth >= 0 and self.forward_smooth <= 1
        assert self.backward_smooth >= 0 and self.backward_smooth <= 1
        self.smooth_matrix = self.generate_smooth_matrix()  # generate smooth matrix

        self.memory_sequence = self.generate_sequence()     # generate memory sequence
        self.stimuli = self.generate_stimuli()              # generate stimuli according to memory sequence
        self.current_timestep = 0                           # reset current timestep
        self.current_step_within_item = 0                   # for CTRNN, when there's multiple steps for each item
        self.testing = False                                # false: encoding phase, true: recall phase
        self.not_retrieved = np.ones((self.batch_size, self.vocabulary_num+1), dtype=bool)   # flag for each item, 0 for retrieved, 1 for not retrieved
        # for i in range(self.batch_size):
        #     self.not_retrieved[i][self.memory_sequence[i]] = True
        self.reported_memory = np.zeros(self.batch_size)    # number of reported memories during recall

    def generate_smooth_matrix(self):
        """
        compute smooth matrix
        
        when generating stimuli, multiply smooth matrix with one-hot encoding of memory sequence

        dim of smooth matrix: memory_num * memory_num
        """
        smooth_matrix = np.eye(self.current_memory_num)
        for i in range(self.current_memory_num - 1):
            for j in range(self.current_memory_num - i - 1):
                smooth_matrix[j+i+1][j] = math.pow(self.forward_smooth, i+1)
                smooth_matrix[j][j+i+1] = math.pow(self.backward_smooth, i+1)
        return smooth_matrix

    def generate_sequence(self, batch_size=None):
        """
        generate memory sequence, each item is a number from 1 to vocabulary_num

        output: batch_size * memory_num
        """
        if batch_size is not None:
            self.batch_size = batch_size
        rand_index = np.repeat(np.arange(1, self.vocabulary_num+1).reshape(1, -1), self.batch_size, axis=0)
        for i in range(self.batch_size):
            np.random.shuffle(rand_index[i])
        memory_sequence = rand_index[:, :self.current_memory_num]
        return memory_sequence      # axis 0: batch size, axis 1: length of sequence

    def generate_stimuli(self, batch_size=None):
        """
        generate stimuli according to memory sequence

        when there's no smoothing, turn memory sequence into one-hot encoding

        output: batch_size * memory_num * (vocabulary_num+1)
        """
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.batch_size
        data = np.zeros((batch_size, self.current_memory_num, self.vocabulary_num+1))
        for i in range(batch_size):
            data[i, :, :] = np.eye(self.vocabulary_num+1)[self.memory_sequence[i]]
        self.smooth_matrix = self.generate_smooth_matrix()
        data = np.einsum('jk,ikl->ijl', self.smooth_matrix, data).transpose(2, 0, 1)
        data = data / np.linalg.norm(data, axis=0)
        data = data.transpose(1, 2, 0)
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

        input: action, batch_size * 1
        output: returned_actions, batch_size * (vocabulary_num+1)
        """
        if self.current_timestep == 0 and not self.testing:
            returned_actions = np.zeros((len(action), self.vocabulary_num+1))
        else:
            returned_actions_digit = np.array([a.cpu().detach().item() for a in action])
            returned_actions = np.eye(self.vocabulary_num+1)[returned_actions_digit]
        return returned_actions
    
    def compute_reward(self, action):
        """
        compute rewards according to actions

        if action is in memory_sequence and hasn't been retrieved, give true reward
        if action is in memory_sequence but has been retrieved, give repeat penalty
        if action is not in memory_sequence, give false reward
        if action is 0, give not_know_reward

        input: action, batch_size * 1
        output: rewards, batch_size * 1
        """
        rewards = np.zeros(len(action))
        for i in range(len(action)):
            if action[i] in list(self.memory_sequence[i]):
                if self.not_retrieved[i][action[i]]:
                    rewards[i] = self.true_reward
                    self.not_retrieved[i][action[i]] = False
                else:
                    rewards[i] = self.repeat_penalty
                self.reported_memory[i] += 1
            elif action[i] == 0:
                rewards[i] = self.not_know_reward
            else:
                rewards[i] = self.false_reward
                self.reported_memory[i] += 1
        return rewards
    
    def check_done(self):
        """
        check whether the trial has completed
        """
        if self.current_timestep >= self.retrieve_time_limit or np.min(self.reported_memory) >= self.current_memory_num or np.sum(self.not_retrieved) == 0:
            return True
        else:
            return False
    
    def step(self, action, batch_size=None):
        """
        for reinforcement learning
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        assert batch_size <= self.batch_size
        assert len(action) == batch_size
        start_recall = 0
        # compute returned action, if needed
        if self.return_action:
            returned_actions = self.get_returned_action(action)
        # recall phase, compute rewards
        if self.testing:
            rewards = self.compute_reward(action)
            observations = np.zeros((batch_size, self.vocabulary_num+1))
            info = {"phase": "recall"}
            self.increase_timestep()
            done = self.check_done()
        else:
            rewards = np.zeros(batch_size)
            done = False
            if self.encode_reward_weight > 0:
                for i in range(batch_size):
                    # during encoding, train the model to output the just inputed item
                    if action[i] == self.memory_sequence[i][self.current_timestep]:
                        rewards[i] = self.true_reward * self.encode_reward_weight
                    else:
                        rewards[i] = self.false_reward * self.encode_reward_weight
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
            observations = np.concatenate((observations, np.array([start_recall for _ in range(batch_size)]).reshape(-1, 1)), axis=1)
        if self.return_action:
            if self.testing and self.current_timestep > 0:
                observations = np.concatenate((observations, returned_actions), axis=1)
            else:
                observations = np.concatenate((observations, np.zeros((batch_size, self.vocabulary_num+1))), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, rewards.reshape(-1, 1)), axis=1)
        return observations, rewards, done, info

    def reset(self, regenerate_contexts=True, batch_size=None):
        """
        reset the trial, return the first observation
        """
        if regenerate_contexts:
            self.current_memory_num = np.random.randint(self.memory_num - self.memory_var, self.memory_num + self.memory_var + 1)
            self.memory_sequence = self.generate_sequence(batch_size)
            self.stimuli = self.generate_stimuli()

        self.current_timestep = 0
        self.current_step_within_item = 0
        self.testing = False
        self.not_retrieved = np.ones((self.batch_size, self.vocabulary_num+1), dtype=bool)
        self.reported_memory = np.zeros(self.batch_size)
        info = {"phase": "encoding"}
        observations = self.stimuli[:, self.current_timestep, :]
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.zeros((self.batch_size, 1))), axis=1)
        if self.return_action:
            observations = np.concatenate((observations, np.zeros((self.batch_size, self.vocabulary_num+1))), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, np.zeros((self.batch_size, 1))), axis=1)
        return observations, info

    def render(self, mode='human'):
        pass

    def get_batch(self, batch_size=None):
        """
        for supervised learning, return inputs and ground truth data for all timesteps
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        assert batch_size <= self.batch_size
        data = np.zeros((self.current_memory_num + self.retrieve_time_limit, batch_size, self.vocabulary_num+1))
        gt = np.zeros((self.current_memory_num + self.retrieve_time_limit, batch_size, self.vocabulary_num+1))
        data[:self.current_memory_num, :, :] = self.stimuli.transpose(1, 0, 2)
        for i in range(batch_size):
            gt[self.current_memory_num:self.current_memory_num*2, i, :] = np.eye(self.vocabulary_num+1)[self.memory_sequence[i]]
        if self.steps_each_item > 1:
            data = np.repeat(data, self.steps_each_item, axis=0)
            gt = np.repeat(gt, self.steps_each_item, axis=0)
        if self.start_recall_cue:
            cue = np.zeros((self.current_memory_num + self.retrieve_time_limit, batch_size, 1))
            cue[self.current_memory_num, :, 0] = 1
            data = np.concatenate((data, cue), axis=2)
        return data, gt

    def compute_accuracy(self, actions):
        """
        compute accuracy for all timesteps

        input: actions, retrieve_time_limit * batch_size
        outputs: number of three types of results: correct, wrong, not_know, i.e. 3 int
        """
        assert len(actions[0]) <= self.batch_size
        batch_size = min(len(actions[0]), self.batch_size)
        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0
        not_retrieved = np.ones((batch_size, self.vocabulary_num+1), dtype=bool)
        for actions_batch in actions[self.current_memory_num:]:
            for i, action in enumerate(actions_batch):
                if action in list(self.memory_sequence[i]) and not_retrieved[i][action]:
                    correct_actions += 1
                    not_retrieved[i][action] = False
                elif action == 0:
                    not_know_actions += 1
                else:
                    wrong_actions += 1
        return correct_actions, wrong_actions, not_know_actions

    def compute_rewards(self, actions):
        """
        compute rewards for all timesteps
        
        input: actions, timesteps * batch_size
        output: rewards, batch_size * timesteps
        """
        assert len(actions[0]) <= self.batch_size
        batch_size = min(len(actions[0]), self.batch_size)
        rewards = [[] for _ in range(batch_size)]
        not_retrieved = np.ones((batch_size, self.vocabulary_num+1), dtype=bool)
        for t, action_batch in enumerate(actions[i]):
            for i, action in enumerate(action_batch):
                if t < self.current_memory_num:
                    rewards[i].append(0.0)
                else:
                    if action in list(self.memory_sequence[i]):
                        if not_retrieved[i][action]:
                            rewards[i].append(self.true_reward)
                            not_retrieved[i][action] = False
                        else:
                            rewards[i].append(self.repeat_penalty)
                    elif action == 0:
                        rewards[i].append(self.not_know_reward)
                    else:
                        rewards[i].append(self.false_reward)
        return np.array(rewards).transpose(1, 0)


# test
if __name__ == "__main__":
    env = FreeRecall(batch_size=1, vocabulary_num=10, memory_num=5, retrieve_time_limit=5)


