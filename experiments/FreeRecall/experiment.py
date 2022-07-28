from calendar import c
import numpy as np
import sklearn.metrics.pairwise as pairwise
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

    context_num = len(actions)
    trial_num = len(actions[0])

    # for i in range(context_num):
    #     for j in range(trial_num):
    #         print("context {}, trial {}, action: {}, gt: {}, rewards: {}".format(i, j+1, actions[i][j][env.memory_num:], memory_contexts[i][0], 
    #             rewards[i][j][env.memory_num:]))

    # for i in range(context_num):
    #     readout = readouts[i][0]
    #     similarity = pairwise.cosine_similarity(readouts[i][0]["c"][env.memory_num:, 0], readouts[i][0]["c"][:env.memory_num, 0])
    #     plt.imshow(similarity, cmap="Blues")
    #     plt.colorbar()
    #     plt.xlabel("encoding timestep")
    #     plt.ylabel("recall timestep")
    #     plt.title("stimuli-recall similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[i][0], actions[i][0][env.memory_num:]))
    #     savefig(paths["fig"], "similarity_c_{}".format(i))

    # lstm_states = []
    # for i in range(context_num):
    #     lstm_states.append(readouts[i][0]['c'])
    # lstm_states = np.stack(lstm_states).squeeze()
    # pca = PCA()
    # pca.fit(lstm_states)
    # pca.visualize_state_space(save_path=paths["fig"]/"pca"/"memorizing", end_step=env.memory_num)
    # pca.visualize_state_space(save_path=paths["fig"]/"pca"/"recalling", start_step=env.memory_num)

    # if "em_gate" in readouts[0][0].keys():
    #     # specific analysis for models using episodic memory
    #     for i in range(context_num):
    #         readout = readouts[i][0]
    #         em_gates = readout['em_gate']
    #         plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[env.memory_num:], label="context {}".format(i))
    #     plt.xlabel("timesteps of recalling phase")
    #     plt.ylabel("em_gate")
    #     savefig(paths["fig"], "em_gate")
    #     # plt.show()

    #     for i in range(context_num):
    #         readout = readouts[i][0]
    #         recc_gates = readout['LSTM']['z_f']
    #         # print(recc_gates.shape)
    #         plt.plot(np.mean(recc_gates.squeeze(1), axis=-1)[env.memory_num:], label="context {}".format(i))
    #     plt.xlabel("timesteps of recalling phase")
    #     plt.ylabel("recc_gate")
    #     savefig(paths["fig"], "recc_gate")

    #     ratios = []
    #     for i in range(context_num):
    #         readout = readouts[i][0]
    #         recc_act = readout['LSTM']['rec']
    #         mem_act = readout['memory']
    #         # print(recc_gates.shape)
    #         ratio = np.mean(recc_act.squeeze(1), axis=-1)[env.memory_num:]/np.mean(mem_act.squeeze(1), axis=-1)[env.memory_num:]
    #         plt.plot(ratio, label="context {}".format(i))
    #         ratios.append(ratio)
    #     plt.xlabel("timesteps of recalling phase")
    #     plt.ylabel("recurrent activation : memory activation")
    #     savefig(paths["fig"], "recc_mem_rate")
    #     ratios = np.array(ratios)
    #     print("recurrent activation : memory activation {}".format(np.mean(ratios, axis=0)))

    #     for i in range(context_num):
    #         readout = readouts[i][0]
    #         similarity = readout['ValueMemory']['BasicSimilarity']['similarities']    # x: context, y: timesteps
    #         plt.figure(figsize=(7, 3), dpi=250)
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(similarity, cmap="Blues")
    #         plt.colorbar()
    #         plt.xlabel("memory")
    #         plt.ylabel("timesteps")
    #         plt.subplot(1, 2, 2)
    #         plt.plot(np.argmax(similarity, axis=1))
    #         plt.xlabel("timesteps")
    #         plt.ylabel("chosen memory")
    #         plt.suptitle("memory-query similarity\nmemory sequence {}, recall sequence {}".format(memory_contexts[i][0], actions[i][0][env.memory_num:]))
    #         plt.tight_layout()
    #         savefig(paths["fig"], "similarity_memory_{}".format(i))
        
    #     for i in range(context_num):
    #         readout = readouts[i][0]
    #         similarity = readout['ValueMemory']['BasicSimilarity']['similarities']
    #         plt.plot(np.argmax(similarity, axis=1))
    #     plt.xlabel("timesteps")
    #     plt.ylabel("chosen memory")
    #     plt.tight_layout()
    #     savefig(paths["fig"], "chosen_memory")

        # for i in range(context_num):
        #     readout = readouts[i][0]
        #     similarity = readout['ValueMemory']['BasicSimilarity']['similarities']
        #     plt.plot(memory_contexts[i][0][np.argmax(similarity, axis=1)], label="context {}".format(i))
        # plt.xlabel("timesteps")
        # plt.ylabel("max similarity memory context")
        # plt.legend()
        # savefig(paths["fig"], "max_similarity_memory_context")
        # plt.show()

    """ SVM decoding """
    c_memorizing = np.stack([readouts[i][0]['c'][:env.memory_num].squeeze() for i in range(context_num)]).transpose(1, 0, 2)
    c_recalling = np.stack([readouts[i][0]['c'][env.memory_num:].squeeze() for i in range(context_num)]).transpose(1, 0, 2)
    dec_act_memorizing = np.stack([readouts[i][0]['dec_act'][:env.memory_num].squeeze() for i in range(context_num)]).transpose(1, 0, 2)
    dec_act_recalling = np.stack([readouts[i][0]['dec_act'][env.memory_num:].squeeze() for i in range(context_num)]).transpose(1, 0, 2)
    memories = np.stack([readouts[i][0]['memory'][env.memory_num:].squeeze() for i in range(context_num)]).transpose(1, 0, 2)
    memory_sequence = np.stack([memory_contexts[i][0] for i in range(context_num)]).transpose(1, 0) - 1
    
    svm = SVM()
    svm.fit(c_memorizing, memory_sequence)
    svm.visualize(save_path=paths["fig"]/"svm"/"c_mem")

    svm.fit(c_recalling, memory_sequence)
    svm.visualize(save_path=paths["fig"]/"svm"/"c_rec")

    svm.fit(dec_act_memorizing, memory_sequence)
    svm.visualize(save_path=paths["fig"]/"svm"/"dec_act_mem")

    svm.fit(dec_act_recalling, memory_sequence)
    svm.visualize(save_path=paths["fig"]/"svm"/"dec_act_rec")

    svm.fit(memories, memory_sequence)
    svm.visualize(save_path=paths["fig"]/"svm"/"memory")
