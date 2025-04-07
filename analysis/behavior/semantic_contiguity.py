import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

from utils import savefig


class SemanticContiguity:
    def __init__(self) -> None:
        self.results = None
        self.results_all_time = None

    def fit(self, actions, feature_dim):
        context_num = actions.shape[0]
        memory_num = actions.shape[1]
        feature_num = actions.shape[-1]
        self.results = np.zeros(feature_num+1)
        for i in range(context_num):
            for t in range(memory_num-1):
                if actions[i][t][-1] == 1 or actions[i][t+1][-1] == 1:
                    continue
                similarity = np.sum(actions[i][t][:feature_num] * actions[i][t+1][:feature_num])
                self.results[int(similarity)] += 1
        self.results = self.results / np.sum(self.results)

        self.baseline = np.zeros(feature_num+1)
        for i in range(feature_num+1):
            self.baseline[i] = comb(feature_num, i) * ((1/feature_dim) ** i) * ((1-1/feature_dim) ** (feature_num-i))

        self.results_normalized = self.results / self.baseline
        self.results_normalized = self.results_normalized / np.sum(self.results_normalized)

        return self.results

    def visualize(self, save_path, save_name="all_time", use_normalized=False, title="", format="png"):
        if use_normalized:
            data = self.results_normalized
        else:
            data = self.results

        plt.figure(figsize=(4, 3.3), dpi=180)
        plt.scatter(np.arange(self.results.shape[0]), data, c='b', zorder=2)
        plt.plot(np.arange(self.results.shape[0]), data, c='k', zorder=1)
        plt.xlabel("semantic similarity")
        plt.ylabel("conditional\nrecall probability")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        savefig(save_path, save_name, format=format)

    def get_results(self):
        return self.results
  
