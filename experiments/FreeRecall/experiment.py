import numpy as np
import sklearn.metrics.pairwise as pairwise
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import savefig


def run(data, model, env, paths):
    readouts = data['readouts']
    actions = data['actions']
    probs = data['probs']
    rewards = data['rewards']
    values = data['values']
    memory_contexts = np.array(data['memory_contexts'])

    env.render()

    context_num = len(actions)
    trial_num = len(actions[0])

    for i in range(context_num):
        for j in range(trial_num):
            print("context {}, trial {}, action: {}, gt: {}, rewards: {}".format(i, j+1, actions[i][j][env.memory_num:], memory_contexts[i], 
                rewards[i][j][env.memory_num:]))

    for i in range(context_num):
            readout = readouts[i][0]
            similarity = pairwise.cosine_similarity(readouts[i][0]["c"][env.memory_num:, 0], readouts[i][0]["c"][:env.memory_num, 0])
            plt.imshow(similarity, cmap="Blues")
            plt.colorbar()
            plt.xlabel("encoding timestep")
            plt.ylabel("recall timestep")
            plt.title("stimuli-recall similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[i][0], actions[i][0][env.memory_num:]))
            savefig(paths["fig"], "similarity_c_{}".format(i))

    if "em_gate" in readouts[0][0].keys():
        for i in range(context_num):
            readout = readouts[i][0]
            em_gates = readout['em_gate']
            plt.plot(np.mean(em_gates.squeeze(1), axis=-1), label="context {}".format(i))
        plt.xlabel("timesteps")
        plt.ylabel("em_gate")
        savefig(paths["fig"], "em_gate")
        # plt.show()

        for i in range(context_num):
            readout = readouts[i][0]
            similarity = readout['ValueMemory']['BasicSimilarity']['similarities']    # x: context, y: timesteps
            plt.figure(figsize=(7, 3), dpi=250)
            plt.subplot(1, 2, 1)
            plt.imshow(similarity, cmap="Blues")
            plt.colorbar()
            plt.xlabel("memory")
            plt.ylabel("timesteps")
            plt.subplot(1, 2, 2)
            plt.plot(np.argmax(similarity, axis=1))
            plt.xlabel("timesteps")
            plt.ylabel("chosen memory")
            plt.suptitle("memory-query similarity\nmemory sequence {}, recall sequence {}".format(memory_contexts[i][0], actions[i][0][env.memory_num:]))
            plt.tight_layout()
            savefig(paths["fig"], "similarity_memory_{}".format(i))

        # for i in range(context_num):
        #     readout = readouts[i][0]
        #     similarity = readout['ValueMemory']['BasicSimilarity']['similarities']
        #     plt.plot(memory_contexts[i][0][np.argmax(similarity, axis=1)], label="context {}".format(i))
        # plt.xlabel("timesteps")
        # plt.ylabel("max similarity memory context")
        # plt.legend()
        # savefig(paths["fig"], "max_similarity_memory_context")
        # plt.show()