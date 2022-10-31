import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA, TDR
from analysis.decoding import SVM


def softmax_np(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def run(data_all, model_all, env, paths):
    run_num = len(list(data_all.keys()))
    sim_mem, sim_enc, sim_rec, sim_enc_rec = [], [], [], []
    for run_name, data in data_all.items():
        fig_path = paths["fig"]/run_name

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
                print("context {}, trial {}, gt: {}, action: {}, rewards: {}".format(i, j+1, memory_contexts[i][0][0], actions[i][j][env.memory_num:], 
                    rewards[i][j][env.memory_num:]))

        # f_in imshow
        similarity = readouts[0][0]["ValueMemory"]["similarity"].squeeze()
        print(similarity.shape)
        plt.imshow(similarity, cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recall timestep")
        plt.title("recalling state-memory similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0][0], actions[0][0][env.memory_num:]))
        savefig(fig_path, "similarity_memory")

        similarities = []
        for i in range(context_num):
            similarity = readouts[i][0]["ValueMemory"]["similarity"].squeeze()
            similarities.append(similarity)
        similarities = np.stack(similarities)
        similarity = np.mean(similarities, axis=0)
        plt.imshow(similarity, cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recall timestep")
        plt.title("recalling state-memory similarity\nmean of 20 trials")
        savefig(fig_path, "similarity_memory_mean")
        sim_mem.append(similarity)
        
        plt.imshow(similarity, cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recall timestep")
        plt.title("recalling state-memory similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0][0], actions[0][0][env.memory_num:]))
        savefig(fig_path, "similarity_memory")

        # similarity of states
        states = readouts[0][0]["state"].squeeze()
        print(np.mean(states, axis=1))
        similarity = np.dot(states, states.T)
        for i in range(similarity.shape[0]):
            similarity[i, :env.memory_num] = normalize(similarity[i, :env.memory_num])
            similarity[i, env.memory_num:] = normalize(similarity[i, env.memory_num:])

        plt.imshow(similarity[:env.memory_num, :env.memory_num], cmap="Blues")
        plt.colorbar()
        plt.title("encoding state similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0][0], actions[0][0][env.memory_num:]))
        savefig(fig_path, "similarity_state_encode")
        sim_enc.append(similarity[:env.memory_num, :env.memory_num])

        plt.imshow(similarity[env.memory_num:, env.memory_num:], cmap="Blues")
        plt.colorbar()
        plt.title("recalling state similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0][0], actions[0][0][env.memory_num:]))
        savefig(fig_path, "similarity_state_recall")
        sim_rec.append(similarity[env.memory_num:, env.memory_num:])

        plt.imshow(similarity[env.memory_num:, :env.memory_num], cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recalling timestep")
        plt.title("encoding-recalling state similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0][0], actions[0][0][env.memory_num:]))
        savefig(fig_path, "similarity_state_encode_recall")
        sim_enc_rec.append(similarity[env.memory_num:, :env.memory_num])
    
    sim_mem = np.mean(np.stack(sim_mem), axis=0)
    sim_enc = np.mean(np.stack(sim_enc), axis=0)
    sim_rec = np.mean(np.stack(sim_rec), axis=0)
    sim_enc_rec = np.mean(np.stack(sim_enc_rec), axis=0)

    plt.imshow(sim_mem, cmap="Blues")
    plt.colorbar()
    plt.xlabel("encoding timestep")
    plt.ylabel("recall timestep")
    plt.title("recalling state-memory similarity\nmean of 10 models")
    savefig(paths['fig'], "similarity_memory_mean")

    plt.imshow(sim_enc, cmap="Blues")
    plt.colorbar()
    plt.title("encoding state similarity\nmean of 10 models")
    savefig(paths['fig'], "similarity_state_encode_mean")

    plt.imshow(sim_rec, cmap="Blues")
    plt.colorbar()
    plt.title("recalling state similarity\nmean of 10 models")
    savefig(paths['fig'], "similarity_state_recall_mean")

    plt.imshow(sim_enc_rec, cmap="Blues")
    plt.colorbar()
    plt.xlabel("encoding timestep")
    plt.ylabel("recalling timestep")
    plt.title("encoding-recalling state similarity\nmean of 10 models")
    savefig(paths['fig'], "similarity_state_encode_recall_mean")

        # mem gate
        # plt.figure(figsize=(4, 3), dpi=250)
        # for i in range(context_num):
        #     readout = readouts[i][0]
        #     em_gates = readout['mem_gate_encode']
        #     plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[:env.memory_num], label="context {}".format(i))
        # ax = plt.gca()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.xlabel("timesteps of recalling phase")
        # plt.ylabel("memory gate")
        # plt.tight_layout()
        # savefig(fig_path, "em_gate_encode")

        # plt.figure(figsize=(4, 3), dpi=250)
        # for i in range(context_num):
        #     readout = readouts[i][0]
        #     em_gates = readout['mem_gate_recall']
        #     plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[:env.memory_num], label="context {}".format(i))
        # ax = plt.gca()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.xlabel("timesteps of recalling phase")
        # plt.ylabel("memory gate")
        # plt.tight_layout()
        # savefig(fig_path, "em_gate_recall")

        # # recalled probability
        # recalled_times = np.zeros((env.memory_num, env.memory_num))
        # actions = np.array(actions)
        # for i in range(all_context_num):
        #     for t in range(env.memory_num - 1):
        #         position1 = np.where(memory_contexts[i][0][0] == actions[i][0][env.memory_num+t])
        #         position2 = np.where(memory_contexts[i][0][0] == actions[i][0][env.memory_num+t+1])
        #         if position1[0].shape[0] != 0 and position2[0].shape[0] != 0:
        #             # print(position1[0].shape[0], position2[0].shape[0])
        #             position1 = position1[0][0]
        #             position2 = position2[0][0]
        #             recalled_times[position1][position2] += 1
        # times_sum = np.expand_dims(np.sum(recalled_times, axis=1), axis=1)
        # times_sum[times_sum == 0] = 1
        # recalled_times = recalled_times / times_sum
        # for t in range(env.memory_num):
        #     if t != 0:
        #         plt.scatter(np.arange(1, t+1), recalled_times[t][:t], c='b')
        #     if t != env.memory_num-1:
        #         plt.scatter(np.arange(t+2, env.memory_num+1), recalled_times[t][t+1:], c='b')
        #     plt.scatter(np.array([t+1]), recalled_times[t][t], c='r')
        #     plt.xlabel("item position")
        #     plt.ylabel("possibility of next recalling")
        #     plt.title("current position: {}".format(t+1))
        #     savefig(fig_path/"recall_prob", "timestep_{}".format(t+1))

        # # PCA
        # states = []
        # for i in range(context_num):
        #     states.append(readouts[i][0]['state'])
        # states = np.stack(states).squeeze()
        # pca = PCA()
        # pca.fit(states)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"memorizing", end_step=env.memory_num)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"recalling", start_step=env.memory_num, end_step=env.memory_num*2)
        # pca.visualize_state_space(save_path=fig_path/"pca")

        # half_states = []
        # for i in range(context_num):
        #     half_states.append(readouts[i][0]['half_state'])
        # half_states = np.stack(half_states).squeeze()
        # pca = PCA()
        # pca.fit(half_states)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"half_state"/"memorizing", end_step=env.memory_num)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"half_state"/"recalling", start_step=env.memory_num, end_step=env.memory_num*2)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"half_state")

        # # SVM
        # c_memorizing = np.stack([readouts[i][0]['state'][:env.memory_num].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)
        # c_recalling = np.stack([readouts[i][0]['state'][-env.memory_num:].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)
        # memory_sequence = np.stack([memory_contexts[i][0][0] for i in range(all_context_num)]).transpose(1, 0) - 1
        
        # svm = SVM()
        # svm.fit(c_memorizing, memory_sequence)
        # svm.visualize(save_path=fig_path/"svm"/"c_mem")
        # svm.visualize_by_memory(save_path=fig_path/"svm"/"c_mem")

        # svm.fit(c_recalling, memory_sequence)
        # svm.visualize(save_path=fig_path/"svm"/"c_rec")
        # svm.visualize_by_memory(save_path=fig_path/"svm"/"c_rec")

        # # TDR
        # states = []
        # for i in range(context_num):
        #     states.append(readouts[i][0]['state'][:env.memory_num*2])
        # states = np.stack(states).squeeze(2)
        # tdr = TDR()
        # # order_var_single_trial = np.arange(env.memory_num*2) % env.memory_num
        # order_var_single_trial = np.arange(env.memory_num*2)
        # order_var = np.repeat(order_var_single_trial.reshape(1, -1), context_num, axis=0)
        # item_var = np.zeros((context_num, env.memory_num*2))
        # for i in range(context_num):
        #     item_var[i, :env.memory_num] = memory_contexts[i][0][0]
        #     item_var[i, env.memory_num:env.memory_num*2] = actions[i][0][env.memory_num:]
        # task_var = np.stack((order_var, item_var))
        # task_var = np.transpose(task_var, (1, 2, 0))
        # var_labels = ["index", "item"]

        # tdr.fit(states, task_var, var_labels)
        # tdr.visualize(var_labels=var_labels, save_path=fig_path/"tdr", save_fname="memorizing", end_step=env.memory_num)
        # tdr.visualize(var_labels=var_labels, save_path=fig_path/"tdr", save_fname="recalling", start_step=env.memory_num, end_step=env.memory_num*2)
        # tdr.visualize_coef_norm(save_path=fig_path/"tdr")

        # PCA combining states and half_states
        # states = []
        # for i in range(context_num):
        #     states.append(readouts[i][0]['state'])
        # states = np.stack(states).squeeze()

        # half_states = []
        # for i in range(context_num):
        #     half_states.append(readouts[i][0]['half_state'])
        # half_states = np.stack(half_states).squeeze()

        # all_states = np.concatenate([states, half_states], axis=1)
        # timesteps = states.shape[1]

        # pca = PCA()
        # pca.fit(all_states)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"all_together"/"memorizing", end_step=env.memory_num)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"all_together"/"recalling", start_step=env.memory_num, end_step=env.memory_num*2)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"all_together", end_step=env.memory_num*2)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"all_together"/"half_state"/"memorizing", start_step=timesteps, end_step=timesteps+env.memory_num,
        #     display_start_step=0, display_end_step=env.memory_num)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"all_together"/"half_state"/"recalling", start_step=timesteps+env.memory_num, end_step=timesteps+env.memory_num*2,
        #     display_start_step=env.memory_num, display_end_step=env.memory_num*2)
        # pca.visualize_state_space(save_path=fig_path/"pca"/"all_together"/"half_state", start_step=timesteps, end_step=timesteps+env.memory_num*2,
        #     display_start_step=0, display_end_step=env.memory_num*2)
