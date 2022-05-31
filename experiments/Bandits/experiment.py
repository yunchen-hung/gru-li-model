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

    context_num = len(actions)
    trial_num = len(actions[0])

    for i in range(context_num):
        for j in range(trial_num):
            print("context {}, trial {}, action: {}, rewards: {}".format(i, j+1, actions[i][j], rewards[i][j]))

    print("memory_contexts", memory_contexts)

    for i in range(context_num):
        readout = readouts[i][0]
        em_gates = readout['em_gate']
        plt.plot(em_gates.squeeze(), label="context {}".format(i))
    plt.legend()
    plt.xlabel("timesteps")
    plt.ylabel("em_gate")
    savefig(paths["fig"], "em_gate")
    # plt.show()

    for i in range(context_num):
        readout = readouts[i][0]
        similarity = readout['ValueMemory']['LCASimilarity']['similarities']    # x: context, y: timesteps
        plt.imshow(similarity, cmap="Blues")
        plt.colorbar()
        plt.xlabel("memory")
        plt.ylabel("timesteps")
        plt.title("memory-query similarity, context {}".format(i))
        savefig(paths["fig"], "similarity_context_{}".format(i))

        lcas = readout['ValueMemory']['LCASimilarity']['lcas'][:, -1]
        plt.imshow(lcas, cmap="Blues")
        plt.colorbar()
        plt.xlabel("memory")
        plt.ylabel("timesteps")
        plt.title("memory-query LCA similarity, context {}".format(i))
        savefig(paths["fig"], "lcas_context_{}".format(i))

    for i in range(context_num):
        readout = readouts[i][0]
        similarity = readout['ValueMemory']['LCASimilarity']['similarities']
        plt.plot()
        plt.plot(memory_contexts[np.argmax(similarity, axis=1)], label="context {}".format(i))
    plt.xlabel("timesteps")
    plt.ylabel("max similarity memory context")
    plt.legend()
    savefig(paths["fig"], "max_similarity_memory_context")
    # plt.show()

    for i in range(context_num):
        readout = readouts[i][0]
        lcas = readout['ValueMemory']['LCASimilarity']['lcas'][:, -1]
        plt.plot()
        plt.plot(memory_contexts[np.argmax(lcas, axis=1)], label="context {}".format(i))
    plt.xlabel("timesteps")
    plt.ylabel("max LCA similarity memory context")
    plt.legend()
    savefig(paths["fig"], "max_lca_memory_context")