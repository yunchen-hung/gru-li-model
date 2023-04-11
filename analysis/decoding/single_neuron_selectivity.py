import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from utils import savefig


class SingleNeuronSelectivity:
    def __init__(self, n_splits=5) -> None:
        self.n_splits = n_splits

    def fit(self, dataset: np.ndarray, labels: dict):
        self.dt = 1
        act = dataset
        if len(act.shape) == 3:
            X = act.transpose(1, 0, 2).reshape(act.shape[0] * act.shape[1], -1)
        else:
            X = act

        self.label_name = list(labels.keys())
        self.selectivity = np.zeros((len(self.label_name), X.shape[-1]))
        label_idx = 0
        for label_name, label in labels.items():
            label = label.reshape(-1, label.shape[-1])
            # label = label.reshape(-1)
            for i in range(X.shape[-1]):
                data = X[:, i].reshape(-1, 1)
                gt = label
                kf = KFold(n_splits=self.n_splits, shuffle=True)
                selectivity = 0.0
                for train_index, test_index in kf.split(data):
                    data_train, data_test = data[train_index], data[test_index]
                    gt_train, gt_test = gt[train_index], gt[test_index]
                    reg = LinearRegression()
                    reg.fit(data_train, gt_train)
                    # pred = decoder.predict(data_test)
                    # decode_accuracy += np.sum(pred == gt_test) / gt_test.shape[0]
                    selectivity += reg.score(data_test, gt_test)
                selectivity /= self.n_splits
                self.selectivity[label_idx, i] = selectivity
            label_idx += 1
        return self.selectivity
    
    def visualize(self, save_path, file_name="pc_selectivity", title="", pdf=False):
        pass
        # plt.figure(figsize=(4, 3.5), dpi=180)
        # for label_name, data in self.selectivity.items():
        #     plt.plot(np.arange(1, self.n_components+1), data, label=label_name)
        # plt.xlabel("PC")
        # plt.ylabel("Regression R2 score")
        # plt.legend()
        # savefig(save_path, file_name, pdf)
