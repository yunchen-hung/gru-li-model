import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import SVM


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
            print("context {}, trial {}, gt: {}, action: {}, rewards: {}".format(i, j+1, memory_contexts[i][0], actions[i][j][env.memory_num:], 
                rewards[i][j][env.memory_num:]))

    # f_in and recalled item of context 0
    recalled_items = []
    for i in range(env.memory_num):
        # item = np.where(memory_contexts[0][0] == actions[0][0][env.memory_num+i])[0]
        item = np.where(memory_contexts[0][0] == readouts[0][0]["retrieved_idx"][i])[0]
        if item.shape[0] != 0:
            item = item[0]
            recalled_items.append(item)
        f = readouts[0][0]["f_in"][i]
        plt.plot(f[memory_contexts[0][0]], zorder=1, label="context-memory similarity")
        if len(recalled_items) > 1:
            plt.scatter(np.array(recalled_items[:-1]), np.zeros(len(recalled_items)-1), zorder=2, c='g', s=100, label="previously recalled items")
        if len(recalled_items) > 0:
            plt.scatter(np.array(recalled_items[-1]), np.zeros(1), zorder=2, c='r', s=100, label="just-recalled items")
        savefig(paths["fig"]/"recall_behavior", "timestep_{}".format(i))

    # f_in imshow
    similarity = readouts[0][0]["f_in"][:, memory_contexts[0][0]]
    plt.imshow(similarity, cmap="Blues")
    plt.colorbar()
    plt.xlabel("encoding timestep")
    plt.ylabel("recall timestep")
    plt.title("stimuli-recall similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[i][0], actions[i][0][env.memory_num:]))
    savefig(paths["fig"], "similarity_state")

    # recalled probability
    recalled_times = np.zeros((env.memory_num, env.memory_num))
    actions = np.array(actions)
    for i in range(all_context_num):
        for t in range(env.memory_num - 1):
            position1 = np.where(memory_contexts[i][0] == actions[i][0][env.memory_num+t])
            position2 = np.where(memory_contexts[i][0] == actions[i][0][env.memory_num+t+1])
            if position1[0].shape[0] != 0 and position2[0].shape[0] != 0:
                # print(position1[0].shape[0], position2[0].shape[0])
                position1 = position1[0][0]
                position2 = position2[0][0]
                recalled_times[position1][position2] += 1
    times_sum = np.expand_dims(np.sum(recalled_times, axis=1), axis=1)
    times_sum[times_sum == 0] = 1
    recalled_times = recalled_times / times_sum
    for t in range(env.memory_num):
        if t != 0:
            plt.scatter(np.arange(1, t+1), recalled_times[t][:t], c='b')
        if t != env.memory_num-1:
            plt.scatter(np.arange(t+2, env.memory_num+1), recalled_times[t][t+1:], c='b')
        plt.scatter(np.array([t+1]), recalled_times[t][t], c='r')
        plt.xlabel("item position")
        plt.ylabel("possibility of next recalling")
        plt.title("current position: {}".format(t+1))
        savefig(paths["fig"]/"recall_prob", "timestep_{}".format(t+1))

    # PCA
    lstm_states = []
    for i in range(context_num):
        lstm_states.append(readouts[i][0]['state'])
    lstm_states = np.stack(lstm_states).squeeze()
    pca = PCA()
    pca.fit(lstm_states)
    pca.visualize_state_space(save_path=paths["fig"]/"pca"/"memorizing", end_step=env.memory_num)
    pca.visualize_state_space(save_path=paths["fig"]/"pca"/"recalling", start_step=env.memory_num)
    pca.visualize_state_space(save_path=paths["fig"]/"pca")

    # SVM
    c_memorizing = np.stack([readouts[i][0]['state'][:env.memory_num].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)
    c_recalling = np.stack([readouts[i][0]['state'][env.memory_num:].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)
    memory_sequence = np.stack([memory_contexts[i][0] for i in range(all_context_num)]).transpose(1, 0) - 1
    
    svm = SVM()
    svm.fit(c_memorizing, memory_sequence)
    svm.visualize(save_path=paths["fig"]/"svm"/"c_mem")

    svm.fit(c_recalling, memory_sequence)
    svm.visualize(save_path=paths["fig"]/"svm"/"c_rec")
