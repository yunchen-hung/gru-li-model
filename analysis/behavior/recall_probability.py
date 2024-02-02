import numpy as np
import matplotlib.pyplot as plt

from utils import savefig


class RecallProbability:
    def __init__(self) -> None:
        self.results = None
        self.results_all_time = None

    def fit(self, memory_contexts, actions):
        self.context_num, self.memory_num = memory_contexts.shape
        self.results = np.zeros((self.memory_num, self.memory_num))
        for i in range(self.context_num):
            for t in range(self.memory_num - 1):
                position1 = np.where(memory_contexts[i] == actions[i][t])
                position2 = np.where(memory_contexts[i] == actions[i][t+1])
                if position1[0].shape[0] != 0 and position2[0].shape[0] != 0:
                    position1 = position1[0][0]
                    position2 = position2[0][0]
                    self.results[position1][position2] += 1

        self.forward_asymmetry = np.sum(np.triu(self.results, k=1)) / np.sum(self.results)

        times_sum = np.expand_dims(np.sum(self.results, axis=1), axis=1)
        times_sum[times_sum == 0] = 1
        self.results = self.results / times_sum

        self.results_all_time = np.zeros((self.memory_num, self.memory_num*2-1))
        for i in range(self.memory_num):
            self.results_all_time[i, self.memory_num-1-i:self.memory_num*2-1-i] = self.results[i]
        self.results_all_time = np.sum(self.results_all_time, axis=0)
        self.average_times = np.concatenate((np.arange(1, self.memory_num+1),np.arange(self.memory_num-1, 0, -1)), axis=0)
        self.results_all_time = self.results_all_time / self.average_times
        # self.results_all_time = self.results_all_time / np.sum(self.results_all_time)

    def visualize(self, save_path, timesteps=None, title="", format="png"):
        """
        timestep: a list of int indicating the time steps to plot
        """
        if self.results is None or self.results_all_time is None:
            raise Exception("Please run fit() first")
        if timesteps is None:
            timesteps = range(self.memory_num)      # plot at each time step
        for t in timesteps:
            plt.figure(figsize=(5, 4.2), dpi=180)
            if t != 0:
                plt.scatter(np.arange(1, t+1), self.results[t][:t], c='b', zorder=2)
                plt.plot(np.arange(1, t+1), self.results[t][:t], c='k', zorder=1)
            if t != self.memory_num-1:
                plt.scatter(np.arange(t+2, self.memory_num+1), self.results[t][t+1:], c='b', zorder=2)
                plt.plot(np.arange(t+2, self.memory_num+1), self.results[t][t+1:], c='k', zorder=1)
            plt.scatter(np.array([t+1]), self.results[t][t], c='r')
            plt.xlabel("item position")
            plt.ylabel("possibility of next recalling")
            plt.title("current position: {}".format(t+1))

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            savefig(save_path, "timestep_{}".format(t+1), format=format)

    def visualize_all_time(self, save_path, title="", format="png"):
        # plot of all time
        plt.figure(figsize=(5, 4.2), dpi=180)
        plt.scatter(np.arange(-self.memory_num+1, 0), self.results_all_time[:self.memory_num-1], c='b', zorder=2)
        plt.plot(np.arange(-self.memory_num+1, 0), self.results_all_time[:self.memory_num-1], c='k', zorder=1)
        plt.scatter(np.arange(1, self.memory_num), self.results_all_time[self.memory_num:], c='b', zorder=2)
        plt.plot(np.arange(1, self.memory_num), self.results_all_time[self.memory_num:], c='k', zorder=1)
        plt.scatter(np.array([0]), self.results_all_time[self.memory_num-1], c='r')
        plt.xlabel("item position")
        plt.ylabel("conditional recall probability")
        # title = title if title else "conditional recall probability"
        # plt.title(title)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        savefig(save_path, "all_time", format=format)

    def visualize_mat(self, save_path, format="png"):
        # plot of matrix
        plt.figure(figsize=(5, 4.2), dpi=180)
        plt.imshow(self.results, cmap="Blues")
        plt.colorbar()
        plt.xlabel("item position")
        plt.ylabel("recalling timestep")
        plt.title("recalling probability matrix")
        plt.tight_layout()
        savefig(save_path, "recall_prob_mat", format=format)

    def get_results(self):
        return self.results

    def get_results_all_time(self):
        return self.results_all_time

    def set_results(self, results, results_all_time):
        self.results = results
        self.results_all_time = results_all_time
        self.memory_num = self.results.shape[0]


class RecallProbabilityInTime:
    """
    calculate the recall probability of each item at each time step.
    """
    def __init__(self) -> None:
        self.results = None

    def fit(self, memory_contexts, actions, condition=None):
        """
        condition: a tuple (i, t) with 2 int indicating "given recalling ith item at timestep t"
            or an int i indicating "given recalling ith item at timestep i"
        """
        if condition is not None:
            if isinstance(condition, tuple):
                i_cond, t_cond = condition
            elif isinstance(condition, int):
                i_cond = condition
                t_cond = condition
            else:
                raise Exception("condition should be a tuple or an int")
            
            # filter the data with condition
            valid_contexts = []
            for i in range(memory_contexts.shape[0]):
                if actions[i][t_cond] == memory_contexts[i][i_cond]:
                    valid_contexts.append(i)

            if len(valid_contexts) == 0:
                print("No valid data for condition: {}".format(condition))
                return None

            memory_contexts = memory_contexts[valid_contexts]
            actions = actions[valid_contexts]

        self.context_num, self.memory_num = memory_contexts.shape
        self.results = np.zeros((self.memory_num, self.memory_num))
        for i in range(self.context_num):
            for t in range(self.memory_num):
                # position1 = np.where(memory_contexts[i] == actions[i][t])
                position1 = np.where(actions[i] == memory_contexts[i][t])
                if position1[0].shape[0] != 0:
                    position1 = position1[0][0]
                    self.results[t][position1] += 1
        times_sum = np.expand_dims(np.sum(self.results, axis=1), axis=1)
        times_sum[times_sum == 0] = 1
        self.results = self.results / times_sum
        return self.results

    def visualize(self, save_path, timesteps=[0], save_name="output_probability", format="png"):
        for t in timesteps:
            plt.figure(figsize=(5, 4.2), dpi=180)
            plt.scatter(np.arange(1, self.memory_num+1), self.results[t], c='b', zorder=2)
            plt.plot(np.arange(1, self.memory_num+1), self.results[t], c='k', zorder=1)
            plt.xlabel("item position")
            plt.ylabel("output probability")
            plt.title("output probability at timestep {}".format(t+1))
            plt.tight_layout()
            savefig(save_path, save_name+"_timestep{}".format(t), format=format)

    def visualize_mat(self, save_path, save_name="output_probability_by_time_mat", title="output_probability_by_time", format="png"):
        if self.results is None:
            raise Exception("Please run fit() first")
        plt.figure(figsize=(5, 4.2), dpi=180)
        plt.imshow(self.results, cmap="Blues")
        plt.colorbar(label="recall probability")
        plt.xlabel("item position in sequence")
        plt.ylabel("recalling timestep")
        # plt.title("output probability by time")
        plt.tight_layout()
        savefig(save_path, save_name, format=format)
        
