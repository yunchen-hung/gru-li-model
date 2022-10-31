import numpy as np
import sklearn.decomposition as decomposition
import matplotlib.pyplot as plt

from utils import savefig


def plot_state_space(data, save_path=None, save_fname=None, show_3d=False, pdf=False, start_step=None, end_step=None, display_start_step=None, display_end_step=None,
    axis_labels=None, title=None):
    """
    data: n_trials x n_steps x n_var
    """
    n_traj, n_steps, n_var = data.shape

    if start_step is None:
            start_step = 0
    if end_step is None:
        end_step = n_steps
    n_steps = end_step - start_step
    colors = np.array([plt.cm.rainbow.reversed()(i) for i in np.linspace(0, 1, n_steps)])

    plt.figure(figsize=(3.5, 3), dpi=180)
    ax = plt.axes(projection='3d') if show_3d else plt.gca()

    proj_act = data
    for i in range(n_traj):
        if not show_3d:
            ax.plot(proj_act[i, start_step:end_step, 0], proj_act[i, start_step:end_step, 1], color="grey", zorder=1)
        else:
            ax.plot3D(proj_act[i, start_step:end_step, 0], proj_act[i, start_step:end_step, 1], proj_act[i, start_step:end_step, 2], color="grey")
    for i in range(start_step, end_step):
        if not show_3d:
            ax.scatter(proj_act[:, i, 0], proj_act[:, i, 1], s=10, marker='o', facecolors='none', edgecolors=colors[i-start_step], zorder=2)
        else:
            ax.scatter3D(proj_act[:, i, 0], proj_act[:, i, 1], proj_act[:, i, 2], s=10, marker='o', facecolors='none', edgecolors=colors[i-start_step])

    max_x = np.max(proj_act[:, :, 0])
    min_x = np.min(proj_act[:, :, 0])
    max_y = np.max(proj_act[:, :, 1])
    min_y = np.min(proj_act[:, :, 1])
    plt.xlim(min_x - 0.5, max_x + 0.5)
    plt.ylim(min_y - 0.5, max_y + 0.5)

    axis_labels = ["PC 1", "PC 2", "PC 3"] if axis_labels is None else axis_labels
    assert len(axis_labels) > 1
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if show_3d:
        assert len(axis_labels) > 2
        ax.set_zlabel(axis_labels[2])

    display_start_step = display_start_step if display_start_step is not None else start_step
    display_end_step = display_end_step if display_end_step is not None else end_step
    cmap = plt.cm.rainbow.reversed()
    norm = plt.Normalize(display_start_step+1, display_end_step)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="timesteps")
    cb.set_ticks(np.arange(display_start_step+1, display_end_step+1))

    if title:
        plt.title(title)

    # handles = []
    # handles.append(mpatches.Patch(color=colors[0], label="first timestep"))
    # handles.append(mpatches.Patch(color=colors[-1], label="last timestep"))
    # plt.legend(handles=handles, frameon=False)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path and save_fname:
        savefig(save_path, save_fname, pdf=pdf)

