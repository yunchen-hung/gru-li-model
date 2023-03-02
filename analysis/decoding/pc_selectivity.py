import numpy as np
import sklearn.decomposition as decomposition
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from utils import savefig


class PCSelectivity:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.pca = decomposition.PCA(n_components=n_components)
        self.reg = LinearRegression()

    def fit(self, dataset: np.ndarray, labels: dict):
        self.dt = 1
        act = dataset
        if len(act.shape) == 3:
            self.n_steps, self.n_traj = act.shape[1], act.shape[0]
            X = act.reshape(act.shape[0] * act.shape[1], -1)
        else:
            self.n_steps, self.n_traj = act.shape[0], 1
            X = act
        self.proj_act = self.pca.fit_transform(X)

        self.selectivity = {}
        for label_name, label in labels.items():
            self.selectivity[label_name] = []
            for i in range(self.n_components):
                self.selectivity.append(self.reg.score(self.proj_act[:, :i+1], label))
    
    def visualize(self, save_path, title="", pdf=False):
        plt.figure(figsize=(4, 3.5), dpi=180)
        for label_name, data in self.selectivity.items():
            plt.plot(np.arange(1, self.n_components+1), data, label=label_name)
        plt.xlabel("PC")
        plt.ylabel("Regression R2 score")
        plt.legend()
        savefig(save_path, "pc_selectivity", pdf)    
