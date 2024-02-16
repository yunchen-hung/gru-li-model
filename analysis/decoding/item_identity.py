import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from utils import savefig


class ItemIdentityDecoder:
    def __init__(self, decoder=svm.SVC(decision_function_shape='ovo'), n_splits=5):
        self.n_splits = n_splits
        self.decoder = decoder
        self.results = None

    def fit(self, all_data, gts, mask=None):
        """
        data: features, list, can have multiple groups of data, time * context_num * state_dim
        gts: ground truths, list, each item is a vector representing a group of labels, 
             can have multiple groups of labels, time * context_num
        mask: same shape as gts, indicating which data points are valid, time * context_num
        return: decoding accuracy of each group of labels
        """
        results = []
        if mask is None:
            mask = np.ones_like(gts, dtype=bool)
        for i in range(all_data.shape[0]):
            data_i = all_data[i]
            data_accuracy = []
            for j in range(gts.shape[0]):
                if not mask[j][i].any():
                    data_accuracy.append(0.0)
                    continue
                
                gt = gts[j][mask[j]]
                data = data_i[mask[j]]
                # print(i, j, mask[j].shape, gt.shape, data.shape, mask[j])
                kf = KFold(n_splits=self.n_splits, shuffle=True)
                decode_accuracy = 0.0
                for train_index, test_index in kf.split(data):
                    data_train, data_test = data[train_index], data[test_index]
                    gt_train, gt_test = gt[train_index], gt[test_index]
                    self.decoder.fit(data_train, gt_train)
                    pred = self.decoder.predict(data_test)
                    decode_accuracy += np.sum(pred == gt_test) / gt_test.shape[0]
                decode_accuracy /= self.n_splits
                data_accuracy.append(decode_accuracy)
            results.append(data_accuracy)
        self.results = np.array(results)
        return results

    def visualize(self, save_path, format="png"):
        if self.results is None:
            raise Exception("Please run fit() first")
        plt.figure(figsize=(1.0 * self.results.shape[0], 4.2), dpi=180)
        for i, result in enumerate(self.results):
            plt.plot(np.arange(1, self.results.shape[1]+1), result, label="timestep {}".format(i+1))
        plt.legend()
        plt.xlim(0.5, 5.5)
        plt.title("SVM decoding accuracy")
        plt.xlabel("memories")
        plt.ylabel("decoding accuracy")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, "svm_accuracy", format=format)

    def visualize_by_memory(self, save_path, save_name="item_identity_decoding", title=None, xlabel="timesteps", colormap_label="timesteps", format="png"):
        if self.results is None:
            raise Exception("Please run fit() first")
        plt.figure(figsize=(0.6 * self.results.shape[1], 3.3), dpi=180)
        n_steps = self.results.shape[0]
        # colors = sns.color_palette("Spectral", n_steps+1)
        # colors = ["#184E77", "#1A759F", "#168AAD", "#34A0A4", "#52B69A", "#76C893", "#99D98C", "#B5E48C"]
        colors = ["#E76F51", "#EE8959", "#F4A261", "#E9C46A", "#8AB17D", "#2A9D8F", "#287271", "#264653"]
        for i in range(self.results.shape[1]):
            plt.plot(np.arange(1, self.results.shape[0]+1), self.results[:, i], label="item {}".format(i+1), color=colors[i])
        plt.xlim(0.5, 0.5 + self.results.shape[0])
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("item identity\ndecoding accuracy")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        cmap = ListedColormap(colors[:-1])
        norm = plt.Normalize(vmin=0.5, vmax=0.5+n_steps)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, n_steps+1), label=colormap_label, ax=ax)

        if title:
            plt.title(title)

        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, save_name, format=format)
