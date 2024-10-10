import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from utils import savefig


class DMAnswerDecoder:
    """
    decode the answer to a DM task from each timestep of the data
    """
    def __init__(self, decoder=svm.SVC(decision_function_shape='ovo'), n_splits=5):
        self.n_splits = n_splits
        self.decoder = decoder
        self.results = None

    def fit(self, all_data, gt, mask=None):
        """
        data: features, list, can have multiple groups of data, (time * trial_num * state_dim)
        gt: ground truths, list, each item is a vector representing a group of labels, 
             can have multiple groups of labels, (trial_num)
        mask: same shape as gts, indicating which data points are valid, time * context_num
        return: decoding accuracy of each timestep of data, (time)
        """
        results = []
        gt = np.array(gt)
        if mask is None:
            mask = np.ones_like(gt, dtype=bool)
        for i in range(all_data.shape[0]):
            data_i = all_data[i]
  
            # print(j, mask.shape, data_i.shape)
            data = data_i[mask]
            # print(i, data_i.shape,data.shape, mask)
            kf = KFold(n_splits=self.n_splits, shuffle=True)
            decode_accuracy = 0.0
            for train_index, test_index in kf.split(data):
                data_train, data_test = data[train_index], data[test_index]
                gt_train, gt_test = gt[train_index], gt[test_index]
                self.decoder.fit(data_train, gt_train)
                pred = self.decoder.predict(data_test)
                decode_accuracy += np.sum(pred == gt_test) / gt_test.shape[0]
            decode_accuracy /= self.n_splits
            results.append(decode_accuracy)
        self.results = np.array(results)
        return results

    def visualize(self, save_path, save_name="answer_decoding", title=None, 
                    xlabel="timesteps", figsize=None, format="png"):
        if self.results is None:
            raise Exception("Please run fit() first")
        figsize = figsize if figsize is not None else (0.6 * self.results.shape[1], 3.3)
        plt.figure(figsize=figsize, dpi=180)
        n_steps = self.results.shape[0]
        # colors = ["#184E77", "#1A759F", "#168AAD", "#34A0A4", "#52B69A", "#76C893", "#99D98C", "#B5E48C"]
        # colors = ["#E76F51", "#EE8959", "#F4A261", "#E9C46A", "#8AB17D", "#2A9D8F", "#287271", "#264653"]
        plt.plot(np.arange(1, self.results.shape[0]+1), self.results)
        plt.xlim(0.5, 0.5 + self.results.shape[0])
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("answer decoding accuracy")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if title:
            plt.title(title)

        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, save_name, format=format)
