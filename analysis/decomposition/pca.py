import numpy as np
import sklearn.decomposition as decomposition
import matplotlib.pyplot as plt

from utils import savefig


class PCA:
    def __init__(self, n_components=2):
        # super().__init__()
        self.n_components = n_components
        self.pca = decomposition.PCA(n_components=n_components)

    def fit(self, dataset: np.ndarray):
        """
        dataset: n_trials x n_steps x n_neurons
        """
        self.dt = 1
        act = dataset
        if len(act.shape) == 3:
            self.n_steps, self.n_traj = act.shape[1], act.shape[0]
            X = act.reshape(act.shape[0] * act.shape[1], -1)
        else:
            self.n_steps, self.n_traj = act.shape[0], 1
            X = act
        self.proj_act = self.pca.fit_transform(X).reshape(self.n_traj, self.n_steps, -1)
        print("PCA explained variance: ", self.pca.explained_variance_ratio_)
        return self

    def fit_transform(self, dataset):
        self.fit(dataset)
        return np.transpose(self.proj_act, axes=(1, 0, 2))

    def visualize(self, save_path=None, ax=None, x_labels=None, title="", n_components=None, pdf=False):
        n_components = n_components if n_components is not None else self.n_components
        assert n_components <= self.n_components
        t = np.arange(self.n_steps) * self.dt if x_labels is None else x_labels

        plt.figure(figsize=(2.3, 1.5 * n_components), dpi=180) if ax is None else None
        proj_act = self.proj_act
        for component in range(n_components):
            ax = plt.subplot(n_components, 1, component + 1)
            for i in range(self.n_traj):
                plt.plot(t, proj_act[i, :, component])

            plt.ylabel("PC{}".format(component + 1))
            plt.xlabel("timesteps") if component + 1 == n_components else None
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False) if component + 1 < n_components else None
            plt.xticks([]) if component + 1 < n_components else None
            plt.yticks([])
        plt.suptitle(title)
        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, "pca_temporal", pdf=pdf)
        # plot_capture_var(self.pca.explained_variance_ratio_, save_path=save_path, pdf=pdf)

    def visualize_state_space(self, save_path=None, show_3d=False, pdf=False, start_step=None, end_step=None, display_start_step=None, 
        display_end_step=None, title=None):
        if start_step is None:
            start_step = 0
        if end_step is None:
            end_step = self.n_steps
        n_steps = end_step - start_step
        colors = np.array([plt.cm.rainbow.reversed()(i) for i in np.linspace(0, 1, n_steps)])

        plt.figure(figsize=(3.5, 3), dpi=180)
        ax = plt.axes(projection='3d') if show_3d else plt.gca()

        proj_act = self.proj_act
        for i in range(self.n_traj):
            if not show_3d:
                ax.plot(proj_act[i, start_step:end_step, 0], proj_act[i, start_step:end_step, 1], color="grey", zorder=1)
            else:
                ax.plot3D(proj_act[i, start_step:end_step, 0], proj_act[i, start_step:end_step, 1], proj_act[i, start_step:end_step, 2], color="grey")
        for i in range(start_step, end_step):
            if not show_3d:
                ax.scatter(proj_act[:, i, 0], proj_act[:, i, 1], s=10, marker='o', facecolors='none', edgecolors=colors[i-start_step], zorder=2)
            else:
                ax.scatter3D(proj_act[:, i, 0], proj_act[:, i, 1], proj_act[:, i, 2], color=colors[i-start_step], marker='o', markerfacecolor='none')

        max_x = np.max(proj_act[:, :, 0])
        min_x = np.min(proj_act[:, :, 0])
        max_y = np.max(proj_act[:, :, 1])
        min_y = np.min(proj_act[:, :, 1])
        plt.xlim(min_x - 0.5, max_x + 0.5)
        plt.ylim(min_y - 0.5, max_y + 0.5)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if show_3d:
            ax.set_zlabel("PC3")

        display_start_step = display_start_step if display_start_step is not None else start_step
        display_end_step = display_end_step if display_end_step is not None else end_step
        cmap = plt.cm.rainbow.reversed()
        norm = plt.Normalize(display_start_step+1, display_end_step)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="timesteps")
        cb.set_ticks(np.arange(display_start_step+1, display_end_step+1))

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if title:
            plt.title("{}, {} trials".format(title, self.n_traj))

        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, "pca_state_space", pdf=pdf)


def plot_capture_var(captured_var, save_path=None, pdf=False):
    plt.figure(figsize=(3, 2))
    plt.plot(np.arange(len(captured_var))+1, captured_var, marker=".")
    plt.ylabel("Captured variance")
    plt.xlabel("Number of PCs")
    savefig(save_path, "captured_var", pdf=pdf)
