import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from utils import savefig


class SVM:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.results = None

    def fit(self, all_data, gts):
        """
        data: features, list, can have multiple groups of data
        gts: ground truths, list, each item is a vector representing a group of labels, 
             can have multiple groups of labels
        return: decoding accuracy of each group of labels
        """
        results = []
        for i in range(all_data.shape[0]):
            data = all_data[i]
            data_accuracy = []
            for j in range(gts.shape[0]):
                gt = gts[j]
                kf = KFold(n_splits=self.n_splits, shuffle=True)
                decode_accuracy = 0.0
                for train_index, test_index in kf.split(data):
                    data_train, data_test = data[train_index], data[test_index]
                    gt_train, gt_test = gt[train_index], gt[test_index]
                    decoder = svm.SVC(decision_function_shape='ovo')
                    decoder.fit(data_train, gt_train)
                    pred = decoder.predict(data_test)
                    decode_accuracy += np.sum(pred == gt_test) / gt_test.shape[0]
                decode_accuracy /= self.n_splits
                data_accuracy.append(decode_accuracy)
            results.append(data_accuracy)
        self.results = results
        return results

    def visualize(self, save_path, pdf=False):
        if self.results is None:
            raise Exception("Please run fit() first")
        plt.figure(figsize=(1.0 * len(self.results[0]), 4.0), dpi=180)
        for i, result in enumerate(self.results):
            plt.plot(result, label="timestep {}".format(i))
        plt.legend()
        plt.title("SVM decoding accuracy")
        plt.xlabel("memories")
        plt.ylabel("decoding accuracy")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, "svm_accuracy", pdf=pdf)
