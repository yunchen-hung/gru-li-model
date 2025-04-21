import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt


class MultiRegressor:
    """
    a multiple linear regression with index and identity as predictors and hidden states as targets
    """
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.regressor = LinearRegression()

    def fit(self, states, index, identity, mask=None):
        """
        states: trials * timesteps * hidden_dim
        index: trials * timesteps
        identity: trials * timesteps
        mask: trials * timesteps
        """
        if mask is None:
            mask = np.ones_like(index, dtype=bool)
        
        # merge the first two dimensions
        states = states.reshape(-1, states.shape[-1])
        index = index.reshape(-1)
        identity = identity.reshape(-1)
        mask = mask.reshape(-1)

        # convert index and identity to one-hot encoding
        index = np.eye(index.max() + 1)[index]
        identity = np.eye(identity.max() + 1)[identity]
        
        data = np.concatenate([index, identity], axis=-1)
        data = data[mask]
        states = states[mask]
        # print("data shape: ", data.shape)
        # print("states shape: ", states.shape)

        kf = KFold(n_splits=self.n_splits, shuffle=True)
        r2_index, r2_identity = [], []
        for train_index, test_index in kf.split(data):
            data_train, data_test = data[train_index], data[test_index]
            states_train, states_test = states[train_index], states[test_index]

            self.regressor.fit(data_train, states_train)
            states_pred = self.regressor.predict(data_test)
        
            ss_total = np.sum((states_test - np.mean(states_test)) ** 2)
            ss_res = np.sum((states_test - states_pred) ** 2)
            # print(self.regressor.coef_.shape, data_test.shape)
            ss_index = np.sum((self.regressor.coef_[:, :index.shape[1]] @ data_test[:, :index.shape[1]].T) ** 2)
            ss_identity = np.sum((self.regressor.coef_[:, index.shape[1]:] @ data_test[:, index.shape[1]:].T) ** 2)
            r2_index.append(ss_index / ss_total)
            r2_identity.append(ss_identity / ss_total)

        return np.mean(r2_index), np.mean(r2_identity)


if __name__ == "__main__":
    pass
