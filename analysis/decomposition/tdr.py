from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

from utils import savefig


class TDR:
    def __init__(self, smooth_std=40, pca_components=12, reg_time_windows=None):
        super().__init__()
        self.smooth_std = smooth_std  # ms
        self.reg_time_windows = reg_time_windows
        self.scaler = StandardScaler()
        self.lin_reg = LinearRegression()
        self.denoising_pca = PCA(n_components=pca_components)

    # TODO: Normalizer
    def _normalize(self, values, normalization_type, axis=0):
        """
        :param values: 2-dim array
        :param normalization_type: str
        :return: normalized values
        """
        if normalization_type == 'zscore':
            # TODO: some neurons have zero activity
            values_std = values.std(axis=axis)
            values_std[values_std == 0] = 1
            norm_values = (values - values.mean(axis=axis)) / values_std
        elif normalization_type == 'max_abs':
            norm_values = values / np.max(np.abs(values), axis=axis)
        else:
            raise NotImplementedError
        return norm_values

    def _agg_regression_vectors(self, Bs, agg_type, time_windows=None):
        """
        :param Bs: (var, time, neuron) array
        :param agg_type: how the regression vectors are aggregated along the time component
        :return: (var, neuron) array
        """
        n_vars, n_steps, n_neurons = Bs.shape
        if agg_type == 'max_norm':
            B = np.zeros((n_vars, n_neurons))
            for nu in range(n_vars):
                Bnorms = np.linalg.norm(Bs[nu], axis=-1)
                B[nu] = Bs[nu, np.argmax(Bnorms)]

        elif agg_type == 'average':
            if time_windows is not None:
                # print("Average")
                B = np.zeros((n_vars, n_neurons))
                masks = list(time_windows.values())
                for i in range(n_vars):
                    # print(i, n_vars, Bs.shape[0], len(masks))
                    # print(Bs[i, masks[i]].shape)
                    B[i] = np.mean(Bs[i, masks[i]], axis=0)
            else:
                B = Bs.mean(axis=1)
        else:
            raise NotImplementedError
        return B

    def fit(self, task_data, data, conditions: list[dict]=None):
        self.conditions = conditions

        r = data  # n_trials x n_steps x n_neurons
        task_var = task_data  # n_trials x n_steps x n_vars
        n_steps, n_trials, n_neurons = r.shape
        n_vars = task_var.shape[-1]

        # z-score neural response neuron by neuron
        r = self._normalize(r.reshape(n_steps*n_trials, -1), "zscore")
        r = r.reshape(n_steps, n_trials, n_neurons)

        # z-score task variables
        task_var = self._normalize(task_var.reshape(n_steps*n_trials, -1), "max_abs")  # or zscore?
        task_var = task_var.reshape(n_steps, n_trials, n_vars)

        # linear regression of neural activity on task vars
        r = np.transpose(r, (1, 0, 2))  # n_trials x n_steps x n_neurons
        task_var = np.transpose(task_var, (1, 0, 2))  # n_trials x n_steps x n_vars
        B = np.zeros((n_vars, n_steps, n_neurons))
        for i in range(n_neurons):
            for t in range(n_steps):
                self.lin_reg.fit(task_var[:, t], r[:, t, i])
                # B[:,t,i] = np.concatenate([self.lin_reg.coef_, [self.lin_reg.intercept_]])
                B[:, t, i] = self.lin_reg.coef_

        # condition-averaged (optional), smoothed, z-score neural response
        if conditions is not None:
            self.n_traj = len(conditions)
            X = dataset.get_activity_by_condition(conditions, average=True)  # n_conditions x n_steps x n_neurons
            X = np.transpose(X, (1, 0, 2)).reshape(n_steps, -1)  # n_steps x n_conditions*n_neurons
        else:
            X = dataset.get_activity()  # n_steps x n_trials x n_neurons
            self.n_traj = X.shape[1]
            X = X.reshape(n_steps, -1)  # n_steps x n_trials*n_neurons

        X = gaussian_filter1d(X, self.smooth_std / dataset.dt, axis=0)
        X = X.reshape(n_steps * self.n_traj, n_neurons)
        X = self.scaler.fit_transform(X)

        # TODO better results if don't do that...
        # de-noising pca
        # X = self.denoising_pca.fit_transform(X)
        # B = self.denoising_pca.transform(B.reshape(-1, n_neurons)).reshape(n_vars, n_steps, -1)
        # B = gaussian_filter1d(B, self.smooth_std/dataset.dt, axis=1)

        self.B_temporal = B

        if self.reg_time_windows:
            print("Average regression vectors ")
            B = self._agg_regression_vectors(B, agg_type='average', time_windows=self.reg_time_windows)
        else:
            # max norm time-independent regression vectors
            B = self._agg_regression_vectors(B, agg_type='max_norm')

        Borth, R = np.linalg.qr(B.T)
        self.B = Borth.T

        self.X = X.reshape(n_steps, self.n_traj, -1)
        self.proj_X = np.tensordot(self.X, self.B, axes=(2, 1))  # n_steps x n_conditions x n_vars

    def visualize(self, var1_label, var2_label, ax=None, cond_kwargs=None):
        """
        :param var1_label:
        :param var2_label:
        :param ax:
        :param cond_kwargs: plot kwargs for each condition
        :return:
        """
        if ax is None:
            plt.figure(figsize=(4, 3), dpi=130)
            ax = plt.gca()

        assert var1_label in self.var_names, "Expected {} to be in {}".format(var1_label, self.var_names)
        assert var2_label in self.var_names, "Expected {} to be in {}".format(var2_label, self.var_names)
        var1_i = np.argwhere(self.var_names == var1_label)[0]
        var2_i = np.argwhere(self.var_names == var2_label)[0]

        for c_idx, c in enumerate(self.conditions):
            kwargs = cond_kwargs[c_idx] if cond_kwargs is not None else {}
            plt.plot(self.proj_X[:, c_idx, var1_i], self.proj_X[:, c_idx, var2_i], label=c, marker=".",
                     **kwargs)

        plt.xlabel(var1_label)
        plt.ylabel(var2_label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    def visualize_coef_norm(self, save_path=None):
        plt.figure(figsize=(4, 3), dpi=130)
        n_vars = self.B_temporal.shape[0]
        for i in range(n_vars):
            norm = np.linalg.norm(self.B_temporal[i], axis=-1)
            plt.plot(norm, label=self.var_names[i])
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend()
        plt.tight_layout()
        savefig(save_path, "coef_norm")


class CtxdmTDR(Analysis):
    def __init__(self, trial_info_labels=None):
        super().__init__()
        self.trial_info_labels = {
            "Choice": "gt_choice",
            "CohMod1": "coh_mod1",
            "CohMod2": "coh_mod2",
            "Context": "context"
        } if trial_info_labels is None else trial_info_labels

        time_windows = {
            "Choice": [600, 900],
            "CohMod1": [250, 450],
            "CohMod2": [350, 550],
            "Context": [100, 900]
        }
        # TODO
        dt = 50
        time = dt*np.arange(15) + 125
        time_masks = {name: np.logical_and(lower < time, time < upper) for name, (lower, upper) in time_windows.items()}
        # self.tdr = TDR(reg_time_windows=time_masks)
        self.tdr = TDR()

    def fit(self, dataset):
        self.dataset = dataset

    def visualize(self, save_path=None, pdf=False):
        self.fit_visualize(self.dataset, save_path=save_path, pdf=pdf)

    def fit_visualize(self, dataset: TrialDataset, save_path=None, pdf=False):
        n_steps = dataset.get_activity().shape[0]
        gt_choice_key, coh_mod1_key, coh_mod2_key, context_key = [self.trial_info_labels[name] for name in
                                                                  ["Choice", "CohMod1", "CohMod2", "Context"]]
        
        # create task dataset
        task_vars = {name: dataset.get_trial_info(label=label) for name, label in self.trial_info_labels.items()}

        # TODO
        # task_vars["Choice*Context"] = task_vars["Choice"]*task_vars["Context"]
        # task_vars["CohMod1*Context"] = task_vars["CohMod1"]*task_vars["Context"]
        # task_vars["CohMod2*Context"] = task_vars["CohMod2"]*task_vars["Context"]

        task_vars_arr = np.stack(list(task_vars.values()), axis=1)
        task_vars_arr = np.repeat(task_vars_arr[np.newaxis, :, :], n_steps, axis=0)

        task_dataset = TrialDataset(task_vars_arr, dataset.get_trial_info(), list(task_vars.keys()))

        # make the plots
        gt_choice, coh_mod1, coh_mod2, context = [dataset.get_trial_info(label=self.trial_info_labels[name])
                                                  for name in ["Choice", "CohMod1", "CohMod2", "Context"]]

        # TODO
        if len(np.unique(coh_mod1)) == 6:
            coh_mod1_colors = ["black", "grey", "lightgrey", "lightgrey", "grey", "black"]
            coh_mod2_colors = ["#3d5a80", "#98c1d9", "#D1E9F0", "#D1E9F0", "#98c1d9", "#3d5a80"]
            coh_mfc = ["white", "white", "white", None, None, None]
        else:
            coh_mod1_colors = [None]*len(np.unique(coh_mod1))
            coh_mod2_colors = [None]*len(np.unique(coh_mod2))
            coh_mfc = [None]*len(np.unique(coh_mod1))

        kwargs_coh1 = []
        kwargs_coh2 = []
        for i, v in enumerate(np.unique(coh_mod1)):
            kwargs_coh1.append({"color": coh_mod1_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})
            kwargs_coh2.append({"color": coh_mod2_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})

        kwargs_coh1_choice = []
        kwargs_coh2_choice = []
        for i, v in enumerate(np.unique(coh_mod1)):
            for j, v2 in enumerate(np.unique(gt_choice)):
                kwargs_coh1_choice.append(
                    {"color": coh_mod1_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})
                kwargs_coh2_choice.append(
                    {"color": coh_mod2_colors[i], "markerfacecolor": coh_mfc[i], "zorder": 1 - np.abs(v)})

        def _context_plots(context):
            assert context == 1 or context == -1
            row = 0 if context == 1 else 1

            if context == 1:
                conditions = [{coh_mod1_key: v, context_key: context} for v in np.unique(coh_mod1)]
                cond_kwargs = kwargs_coh1
            else:
                conditions = [{coh_mod1_key: v, gt_choice_key: v2, context_key: context}
                              for v in np.unique(coh_mod1) for v2 in np.unique(gt_choice)]
                cond_kwargs = kwargs_coh1_choice

            self.tdr.fit(task_dataset, dataset, conditions)

            ax = plt.subplot(2, 3, 3 * row + 1)
            self.tdr.visualize("Choice", "CohMod1", ax=ax, cond_kwargs=cond_kwargs)

            ax = plt.subplot(2, 3, 3 * row + 2, sharey=ax, sharex=ax)
            plt.title("{} context ".format("Mod1" if context == 1 else "Mod2"))
            if context == 1:
                self.tdr.visualize("Choice", "CohMod2", ax=ax, cond_kwargs=cond_kwargs)

                conditions = [{coh_mod2_key: v, gt_choice_key: v2, context_key: context}
                              for v in np.unique(coh_mod2) for v2 in np.unique(gt_choice)]
                cond_kwargs = kwargs_coh2_choice
            else:
                conditions = [{coh_mod2_key: v, context_key: context} for v in np.unique(coh_mod2)]
                cond_kwargs = kwargs_coh2
                self.tdr.fit(task_dataset, dataset, conditions)
                self.tdr.visualize("Choice", "CohMod1", ax=ax, cond_kwargs=cond_kwargs)

            self.tdr.fit(task_dataset, dataset, conditions)
            ax = plt.subplot(2, 3, 3 * row + 3, sharey=ax, sharex=ax)
            self.tdr.visualize("Choice", "CohMod2", ax=ax, cond_kwargs=cond_kwargs)

        plt.figure(figsize=(6, 4), dpi=180)
        _context_plots(1)
        _context_plots(-1)
        savefig(save_path, "tdr_ctxdm", pdf=pdf)

        # self.tdr.fit(task_dataset, dataset)
        # self.tdr.visualize_coef_norm(save_path)
