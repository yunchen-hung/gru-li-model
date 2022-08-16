import numpy as np
import matplotlib.pyplot as plt

from utils import savefig


def run(data, model, env, paths):
    readouts = data['readouts']
    actions = data['actions']
    probs = data['probs']
    rewards = data['rewards']
    values = data['values']
    memory_contexts = np.array(data['memory_contexts'])

    env.render()

    all_context_num = len(actions)
    context_num = min(all_context_num, 20)
    trial_num = len(actions[0])

    for i in range(context_num):
        for j in range(trial_num):
            print("context {}, trial {}, action: {}, gt: {}, rewards: {}".format(i, j+1, actions[i][j][env.memory_num:], memory_contexts[i][0], 
                rewards[i][j][env.memory_num:]))

    recalled_items = []
    for i in range(env.memory_num):
        recalled_items.append(np.where(memory_contexts[0][0] == actions[0][0][env.memory_num+i])[0][0])
        f = readouts[i][0]["f_in"][i]
        plt.plot(f[memory_contexts[i][0]], zorder=1)
        if len(recalled_items) > 1:
            plt.scatter(np.array(recalled_items[:-1]), np.zeros(len(recalled_items)-1), zorder=2, c='g', s=100)
        plt.scatter(np.array(recalled_items[-1]), np.zeros(1), zorder=2, c='r', s=100)
        savefig(paths["fig"], "recall_timestep_{}".format(i))
