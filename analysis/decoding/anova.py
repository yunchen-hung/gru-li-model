import numpy as np
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt


class ANOVA_F:
    def __init__(self):
        pass

    def fit(self, dataset: np.ndarray, labels: np.ndarray):
        self.dt = 1
        act = dataset
        if len(act.shape) == 3:
            self.n_steps, self.n_traj = act.shape[1], act.shape[0]
            X = act.transpose(1, 0, 2).reshape(act.shape[0] * act.shape[1], -1)
        else:
            self.n_steps, self.n_traj = act.shape[0], 1
            X = act

        labels = labels.reshape(-1)

        # self.selectivity = {}
        # for label_name, label in labels.items():
        #     label = label.reshape(-1)
        #     f, p = f_classif(X, label)
        #     self.selectivity[label_name] = f
        f, p = f_classif(X, labels)
        self.selectivity = f

        return f
