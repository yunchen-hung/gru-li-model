import numpy as np
from sklearn.linear_model import RidgeClassifier

from utils import savefig



class DecoderPerturbation:
    def __init__(self, decoder=RidgeClassifier(), n_splits=1, noise_start=0.0, noise_end=1.0, noise_step=0.05):
        self.decoder = decoder
        self.n_splits = n_splits
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.noise_step = noise_step
        self.results = None

    def fit(self, data, gts, noise_start=None, noise_end=None, noise_step=None):
        """
        data: (n_samples, n_features)
        gts: (n_samples, n_targets)
        """
        self.noise_start = noise_start if noise_start is not None else self.noise_start
        self.noise_end = noise_end if noise_end is not None else self.noise_end
        self.noise_step = noise_step if noise_step is not None else self.noise_step
        accuracies = []
        for i in np.arange(self.noise_start, self.noise_end, self.noise_step):
            accuracy = self.record_with_noise(data, gts, noise_proportion=i)
            print(f"noise proportion: {i}, accuracy: {accuracy}")
            accuracies.append(accuracy)
        self.results = accuracies
        return accuracies


    def visualize(self):
        if self.results is None:
            return
