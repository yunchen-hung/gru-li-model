import numpy as np
import sklearn.decomposition as decomposition
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

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
    
    def transform(self, dataset):
        if len(dataset.shape) == 3:
            X = dataset.reshape(dataset.shape[0] * dataset.shape[1], -1)
        else:
            X = dataset
        return self.pca.transform(X).reshape(self.n_traj, self.n_steps, -1)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return np.transpose(self.proj_act, axes=(1, 0, 2))

    def visualize(self, save_path=None, ax=None, x_labels=None, title="", n_components=None, format="png"):
        n_components = n_components if n_components is not None else self.n_components
        assert n_components <= self.n_components
        t = np.arange(self.n_steps) * self.dt if x_labels is None else x_labels

        plt.figure(figsize=(2.3, 1.5 * n_components), dpi=180) if ax is None else None
        # plt.rcParams['font.size'] = 20
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
            savefig(save_path, "pca_temporal", format=format)
        # plot_capture_var(self.pca.explained_variance_ratio_, save_path=save_path, pdf=pdf)

    def visualize_state_space(self, save_path=None, show_3d=False, start_step=None, end_step=None, 
        display_start_step=None, display_end_step=None, constrain_lim=True, title=None, format="png",
        file_name="pca_state_space", colormap_label="timesteps"):
        if start_step is None:
            start_step = 0
        if end_step is None:
            end_step = self.n_steps
        n_steps = end_step - start_step
        # colors = np.array([plt.cm.rainbow.reversed()(i) for i in np.linspace(0, 1, n_steps)])
        # colors = sns.color_palette("Spectral", n_steps+1)
        # colors = ["#184E77", "#1A759F", "#168AAD", "#34A0A4", "#52B69A", "#76C893", "#99D98C", "#B5E48C"]
        colors = ["#E76F51", "#EE8959", "#F4A261", "#E9C46A", "#8AB17D", "#2A9D8F", "#287271", "#264653"]

        plt.figure(figsize=(3.8, 3.3), dpi=180)
        ax = plt.axes(projection='3d') if show_3d else plt.gca()

        proj_act = self.proj_act
        for i in range(self.n_traj):
            if not show_3d:
                ax.plot(proj_act[i, start_step:end_step, 0], proj_act[i, start_step:end_step, 1], color="grey", zorder=1, linewidth=0.7)
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
        if constrain_lim:
            plt.xlim(min_x - 0.5, max_x + 0.5)
            plt.ylim(min_y - 0.5, max_y + 0.5)

        ax.set_xlabel("PC1 of hidden states")
        ax.set_ylabel("PC2 of hidden states")
        if show_3d:
            ax.set_zlabel("PC3")

        # display_start_step = display_start_step if display_start_step is not None else start_step
        # display_end_step = display_end_step if display_end_step is not None else end_step
        # cmap = plt.cm.rainbow.reversed()
        # norm = plt.Normalize(0.5, display_end_step-display_start_step+0.5)
        # cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, display_end_step-display_start_step+1), ax=ax, label=colormap_label)
        cmap = ListedColormap(colors[:-1])
        norm = plt.Normalize(vmin=0.5, vmax=0.5+n_steps)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, n_steps+1), label=colormap_label)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if title:
            # plt.title("{}, {} trials".format(title, self.n_traj))
            plt.title(title)

        plt.tight_layout()
        if save_path is not None:
            savefig(save_path, file_name, format=format)


def plot_capture_var(captured_var, save_path=None, pdf=False):
    plt.figure(figsize=(3, 2))
    plt.plot(np.arange(len(captured_var))+1, captured_var, marker=".")
    plt.ylabel("Captured variance")
    plt.xlabel("Number of PCs")
    savefig(save_path, "captured_var", pdf=pdf)
