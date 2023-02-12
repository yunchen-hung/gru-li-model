import numpy as np
import matplotlib.pyplot as plt

from utils import savefig


class RecallProbability:
    def __init__(self) -> None:
        self.results = None

    def fit(self, memory_contexts, actions):
        self.results = np.zeros((memory_contexts.shape[1], memory_contexts.shape[1]))
        self.context_num, self.memory_num = memory_contexts.shape
        for i in range(self.context_num):
            for t in range(self.memory_num - 1):
                position1 = np.where(memory_contexts[i] == actions[i][t])
                position2 = np.where(memory_contexts[i] == actions[i][t+1])
                if position1[0].shape[0] != 0 and position2[0].shape[0] != 0:
                    position1 = position1[0][0]
                    position2 = position2[0][0]
                    self.results[position1][position2] += 1
        times_sum = np.expand_dims(np.sum(self.results, axis=1), axis=1)
        times_sum[times_sum == 0] = 1
        self.results = self.results / times_sum

    def visualize(self, save_path, pdf=False):
        if self.results is None:
            raise Exception("Please run fit() first")
        for t in range(self.memory_num):
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

            savefig(save_path, "timestep_{}".format(t+1), pdf=pdf)

    def get_results(self):
        return self.results
    
    def set_results(self, results):
        self.results = results
        self.memory_num = self.results.shape[0]


