import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt

from utils import savefig


class CrossClassifier:
    """
    train a classifier to decode the information from data of all timesteps, then test on another dataset
    used for analyzing how consistent the information is across phases (encoding/recall)
    """
    def __init__(self, decoder=svm.SVC(decision_function_shape='ovo')):
        self.decoder = decoder
        self.results = None

    def fit(self, data_train, gts_train, mask=None):
        """
        data: features, list, can have multiple groups of data, context_num * time * state_dim
        gts: ground truths, list, each item is a vector representing a group of labels, 
             can have multiple groups of labels, context_num * time
        mask: same shape as gts, indicating which data points are valid, context_num * time
        return: overall r2 score and explained variance score
        """
        if mask is None:
            mask = np.ones_like(gts_train, dtype=bool)
        data_train, gts_train = data_train[mask], gts_train[mask]
        
        self.decoder.fit(data_train, gts_train)
        pred = self.decoder.predict(data_train)

        r2 = r2_score(gts_train, pred)
        acc = np.sum(pred == gts_train) / gts_train.shape[0]

        return r2, acc


    def score(self, data_test, gts_test, mask=None):
        if mask is None:
            mask = np.ones_like(gts_test, dtype=bool)
        data_test, gts_test = data_test[mask], gts_test[mask]

        pred = self.decoder.predict(data_test)
        r2 = r2_score(gts_test, pred)
        acc = np.sum(pred == gts_test) / gts_test.shape[0]
        return r2, acc