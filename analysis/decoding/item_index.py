import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from utils import savefig


class ItemIndexDecoder:
    def __init__(self, decoder=svm.SVC(decision_function_shape='ovo'), n_splits=5):
        self.n_splits = n_splits
        self.decoder = decoder
        self.results = None

    def fit(self, all_data, gts, mask=None):
        """
        data: features, list, can have multiple groups of data, context_num * time * state_dim
        gts: ground truths, list, each item is a vector representing a group of labels, 
             can have multiple groups of labels, context_num * time
        mask: same shape as gts, indicating which data points are valid, context_num * time
        return: decoding accuracy of each timestep
        """
        results = []
        if mask is None:
            mask = np.ones_like(gts, dtype=bool)
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        decode_accuracy = []
        for train_index, test_index in kf.split(all_data):
            data_train, data_test = all_data[train_index], all_data[test_index]
            gt_train, gt_test = gts[train_index], gts[test_index]

            data_train = data_train[mask[train_index]]
            gt_train = gt_train[mask[train_index]]

            self.decoder.fit(data_train, gt_train)
            
            accuracy = []
            for i in range(gt_test.shape[1]):
                data_test_timestep = data_test[:, i, :][mask[test_index, i]]
                pred = self.decoder.predict(data_test_timestep)
                accuracy.append(np.sum(pred == gt_test[:, i][mask[test_index, i]]) / data_test_timestep.shape[0])
            decode_accuracy.append(accuracy)
        decode_accuracy = np.sum(np.array(decode_accuracy), axis=0) / self.n_splits

        self.results = decode_accuracy
        return decode_accuracy
        
    def visualize(self, save_path, save_name="item_index_decoding", xlabel="timesteps", format="png"):
        if self.results is None:
            raise Exception("Please run fit() first")
        plt.figure(figsize=(0.6 * self.results.shape[0], 3.3), dpi=180)
        plt.plot(np.arange(1, self.results.shape[0]+1), self.results)
        plt.xlim(0.5, self.results.shape[0]+0.5)
        plt.ylim(0.0, 1.05)
        plt.xlabel(xlabel)
        plt.ylabel("item index\ndecoding accuracy")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, save_name, format=format)

