import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt

from utils import savefig


class Classifier:
    """
    decode the item index from data of all timesteps, and compute the accuracy for each timestep
    """
    def __init__(self, decoder=svm.SVC(decision_function_shape='ovo'), n_splits=5):
        self.n_splits = n_splits
        self.decoder = decoder
        self.results = None

    def fit(self, data, gts, mask=None):
        """
        data: features, list, can have multiple groups of data, context_num * time * state_dim
        gts: ground truths, list, each item is a vector representing a group of labels, 
             can have multiple groups of labels, context_num * time
        mask: same shape as gts, indicating which data points are valid, context_num * time
        return: overall r2 score and explained variance score
        """
        if mask is None:
            mask = np.ones_like(gts, dtype=bool)
        data, gts = data[mask], gts[mask]
        
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        r2_all, acc_all = [], []
        for train_index, test_index in kf.split(data):
            data_train, data_test = data[train_index], data[test_index]
            gt_train, gt_test = gts[train_index], gts[test_index]

            self.decoder.fit(data_train, gt_train)
            pred = self.decoder.predict(data_test)

            r2 = r2_score(gt_test, pred)
            r2_all.append(r2)

            acc = np.sum(pred == gt_test) / gt_test.shape[0]
            acc_all.append(acc)

        r2 = np.mean(r2_all)
        acc = np.mean(acc_all)

        return r2, acc

