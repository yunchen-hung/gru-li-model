import numpy as np
import sklearn.decomposition as decomposition
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from utils import savefig


class PCSelectivity:
    def __init__(self, n_components, reg=LinearRegression(), n_splits=5) -> None:
        self.n_components = n_components
        self.n_splits = n_splits
        self.pca = decomposition.PCA(n_components=n_components)
        self.reg = reg
        # self.reg = LinearRegression()

    def fit(self, dataset: np.ndarray, labels: dict):
        self.dt = 1
        act = dataset
        # print(act.shape)
        if len(act.shape) == 3:
            X = act.reshape(act.shape[0] * act.shape[1], -1)
        else:
            X = act
        self.proj_act = self.pca.fit_transform(X)
        self.explained_variance = self.pca.explained_variance_ratio_
        self.explained_variance = np.cumsum(self.explained_variance)
        # self.proj_act = X
        # print(X.shape)

        self.label_name = list(labels.keys())
        self.selectivity = np.zeros((len(self.label_name), self.n_components))
        label_idx = 0
        for _, label in labels.items():
            # label = label.reshape(-1, label.shape[-1])
            # label = label.reshape(-1)
            if len(label.shape) == 3:
                label = label.reshape(-1, label.shape[-1])
            else:
                label = label.reshape(-1)
            for i in range(self.n_components):
                data = self.proj_act[:, :i+1]
                gt = label
                # print(data.shape, gt.shape)
                kf = KFold(n_splits=self.n_splits, shuffle=True)
                selectivity = 0.0
                for train_index, test_index in kf.split(data):
                    data_train, data_test = data[train_index], data[test_index]
                    gt_train, gt_test = gt[train_index], gt[test_index]
                    # reg = LinearRegression()
                    # reg = Ridge()
                    self.reg.fit(data_train, gt_train)
                    pred = self.reg.predict(data_test)
                    decode_accuracy = np.sum(pred == gt_test) / gt_test.shape[0]
                    selectivity += decode_accuracy
                    # selectivity += self.reg.score(data_test, gt_test)
                selectivity /= self.n_splits
                self.selectivity[label_idx, i] = selectivity
            label_idx += 1
        return self.selectivity, self.explained_variance
                # self.reg.fit(self.proj_act[:, :i+1], label)
                # self.selectivity[label_name].append(self.reg.score(self.proj_act[:, :i+1], label))
                # data = self.proj_act[:, :i+1]
                # gt = label
                # # print(data.shape, gt.shape)
                # kf = KFold(n_splits=self.n_splits, shuffle=True)
                # decode_accuracy = 0.0
                # for train_index, test_index in kf.split(data):
                #     data_train, data_test = data[train_index], data[test_index]
                #     gt_train, gt_test = gt[train_index], gt[test_index]
                #     decoder = svm.SVC(decision_function_shape='ovo')
                #     decoder.fit(data_train, gt_train)
                #     pred = decoder.predict(data_test)
                #     decode_accuracy += np.sum(pred == gt_test) / gt_test.shape[0]
                # decode_accuracy /= self.n_splits
                # self.selectivity[label_name].append(decode_accuracy)
    
    def visualize(self, save_path, file_name="pc_selectivity", title="", format="png"):
        plt.figure(figsize=(4, 3.3), dpi=180)
        c = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, label_name in enumerate(self.label_name):
            plt.plot(np.arange(1, self.n_components+1), self.selectivity[i], label=label_name, color=c[i])
            if hasattr(self, "selectivity_error"):
                plt.fill_between(np.arange(1, self.n_components+1), self.selectivity[i] - self.selectivity_error[i],
                                 self.selectivity[i] + self.selectivity_error[i], color=c[i], alpha=0.2)
        plt.plot(np.arange(1, self.n_components+1), self.explained_variance, label="explained variance", color='k')
        if hasattr(self, "explained_variance_error"):
            plt.fill_between(np.arange(1, self.n_components+1), self.explained_variance - self.explained_variance_error,
                             self.explained_variance + self.explained_variance_error, color='k', alpha=0.2)
        plt.xlabel("PC of hidden states")
        plt.ylabel("decoding accuracy")
        if title:
            plt.title(title)
        plt.legend(fontsize=12)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()

        savefig(save_path, file_name, format=format)

    def set_results(self, label_name, selectivity, explained_variance, selectivity_error, explained_variance_error):
        self.label_name = label_name
        self.selectivity = selectivity
        self.explained_variance = explained_variance
        self.selectivity_error = selectivity_error
        self.explained_variance_error = explained_variance_error
