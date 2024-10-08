import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score
import seaborn as sns
import colorcet as cc

from utils import savefig


class ItemIdentityDecoder:
    """
    decode item identity at each timestep from the data at each timestep respectively
    """
    def __init__(self, decoder=svm.SVC(decision_function_shape='ovo'), n_splits=5):
        self.n_splits = n_splits
        self.decoder = decoder
        self.results = None

    def fit(self, all_data, gts, mask=None):
        """
        data: features, list, can have multiple groups of data, time * context_num * state_dim
        gts: ground truths, list, each item is a vector representing a group of labels, 
             can have multiple groups of labels, time * context_num (* label_dim)
        mask: same shape as gts, indicating which data points are valid, time * context_num
        return: decoding accuracy of each group of labels
        """
        results = []
        r2_all = []
        if mask is None:
            # mask = np.ones_like(gts, dtype=bool)
            mask = np.ones((gts.shape[0], gts.shape[1]), dtype=bool)
        for i in range(all_data.shape[0]):
            data_i = all_data[i]
            data_accuracy = []
            data_r2 = []
            for j in range(gts.shape[0]):
                if not mask[j][i].any():
                    data_accuracy.append(0.0)
                    data_r2.append(-1.0)
                    continue
                
                gt = gts[j][mask[j]]
                # print(j, mask.shape, data_i.shape)
                data = data_i[mask[j]]
                # print(i, j, data_i.shape,data.shape, mask[j])
                kf = KFold(n_splits=self.n_splits, shuffle=True)
                decode_accuracy = 0.0
                r2 = 0.0
                for train_index, test_index in kf.split(data):
                    data_train, data_test = data[train_index], data[test_index]
                    gt_train, gt_test = gt[train_index], gt[test_index]
                    self.decoder.fit(data_train, gt_train)
                    pred = self.decoder.predict(data_test)
                    r2 += r2_score(gt_test, pred)
                    if len(gt.shape) == 2:
                        pred = np.max(pred, axis=1)
                        gt_test = np.max(gt_test, axis=1)
                    decode_accuracy += np.sum(pred == gt_test) / gt_test.shape[0]
                decode_accuracy /= self.n_splits
                r2 /= self.n_splits
                data_accuracy.append(decode_accuracy)
                data_r2.append(r2)
            results.append(data_accuracy)
            r2_all.append(data_r2)
        self.results = np.array(results)
        decode_accuracy_mean = np.mean(np.diagonal(self.results))
        decode_accuracy_last = np.mean(np.diagonal(self.results, offset=-1))
        # r2_all = np.array(r2_all)
        # print(r2_all.shape)
        r2 = np.mean(np.diagonal(r2_all))
        r2_last = np.mean(np.diagonal(r2_all, offset=-1))
        stat_results = {
            "acc": decode_accuracy_mean,
            "acc_last": decode_accuracy_last,
            "r2": r2,
            "r2_last": r2_last
        }
        return results, stat_results
    
    def fit_no_crossval(self, all_data, gts, mask=None):
        results = []
        r2_all = []
        if mask is None:
            # mask = np.ones_like(gts, dtype=bool)
            mask = np.ones((gts.shape[0], gts.shape[1]), dtype=bool)
        for i in range(all_data.shape[0]):
            data_i = all_data[i]
            data_accuracy = []
            data_r2 = []
            for j in range(gts.shape[0]):
                if not mask[j][i].any():
                    data_accuracy.append(0.0)
                    data_r2.append(-1.0)
                    continue
                
                gt = gts[j][mask[j]]
                data = data_i[mask[j]]
                train_inx = np.arange(data.shape[0])
                test_inx = np.arange(data.shape[0])
                data_train, data_test = data[train_inx], data[test_inx]
                gt_train, gt_test = gt[train_inx], gt[test_inx]
                self.decoder.fit(data_train, gt_train)
                pred = self.decoder.predict(data_test)
                r2 = r2_score(gt_test, pred)
                if len(gt.shape) == 2:
                    pred = np.max(pred, axis=1)
                    gt_test = np.max(gt_test, axis=1)
                decode_accuracy = np.sum(pred == gt_test) / gt_test.shape[0]
                
                data_accuracy.append(decode_accuracy)
                data_r2.append(r2)
            results.append(data_accuracy)
            r2_all.append(data_r2)
        self.results = np.array(results)
        decode_accuracy_mean = np.mean(np.diagonal(self.results))
        decode_accuracy_last = np.mean(np.diagonal(self.results, offset=-1))
        # r2_all = np.array(r2_all)
        # print(r2_all.shape)
        r2 = np.mean(np.diagonal(r2_all))
        r2_last = np.mean(np.diagonal(r2_all, offset=-1))
        stat_results = {
            "acc": decode_accuracy_mean,
            "acc_last": decode_accuracy_last,
            "r2": r2,
            "r2_last": r2_last
        }
        return results, stat_results

    def visualize(self, save_path, figsize=None, format="png"):
        if self.results is None:
            raise Exception("Please run fit() first")
        figsize = figsize if figsize is not None else (0.65 * self.results.shape[0], 3.3)
        plt.figure(figsize=figsize, dpi=180)
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

    def visualize_by_memory(self, save_path, save_name="item_identity_decoding", title=None, 
                            xlabel="timesteps", colormap_label="timesteps", figsize=None,
                            format="png"):
        if self.results is None:
            raise Exception("Please run fit() first")
        figsize = figsize if figsize is not None else (0.6 * self.results.shape[1], 3.3)
        plt.figure(figsize=figsize, dpi=180)
        n_steps = self.results.shape[0]
        colors = np.array([cc.cm.rainbow.reversed()(i) for i in np.linspace(0, 0.9, n_steps)])
        # colors = sns.color_palette("Spectral", n_steps+1)
        # colors = ["#184E77", "#1A759F", "#168AAD", "#34A0A4", "#52B69A", "#76C893", "#99D98C", "#B5E48C"]
        # colors = ["#E76F51", "#EE8959", "#F4A261", "#E9C46A", "#8AB17D", "#2A9D8F", "#287271", "#264653"]
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
