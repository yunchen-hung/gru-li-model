import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import SVM
from analysis.visualization import RecallProbability
import sklearn.metrics.pairwise as skp


def run(data_all, model_all, env, paths, exp_name):
    run_num = len(list(data_all.keys()))
    sim_mem, sim_enc, sim_rec, sim_enc_rec = [], [], [], []

    sim_mem = {}
    sim_enc = {}
    sim_rec = {}
    sim_enc_rec = {}
    accuracy = {}
    rec_prob = {}
    rec_prob_mem = {}

    run_names_without_num = []
    for run_name in data_all.keys():
        run_name_without_num = run_name.split("-")[0]
        run_names_without_num.append(run_name_without_num)
        if run_name_without_num not in sim_mem.keys():
            sim_mem[run_name_without_num] = []
            sim_enc[run_name_without_num] = []
            sim_rec[run_name_without_num] = []
            sim_enc_rec[run_name_without_num] = []
            accuracy[run_name_without_num] = []
            rec_prob[run_name_without_num] = []
            rec_prob_mem[run_name_without_num] = []

    for run_name, data in data_all.items():
        fig_path = paths["fig"]/run_name
        print()
        print(run_name)
        run_name_without_num = run_name.split("-")[0]

        model = model_all[run_name]
        if hasattr(model, "step_for_each_timestep"):
            step_for_each_timestep = model.step_for_each_timestep
            timestep_each_phase = step_for_each_timestep * env.memory_num
        else:
            step_for_each_timestep = 1
            timestep_each_phase = env.memory_num
        # timestep_each_phase = env.memory_num

        readouts = data['readouts']
        actions = data['actions']
        probs = data['probs']
        rewards = data['rewards']
        values = data['values']
        accuracy[run_name_without_num].append(data['accuracy'])

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)
        trial_num = len(actions[0])

        memory_contexts = np.array(data['memory_contexts'])
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])
        actions = np.array(actions)
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])
        
        for i in range(5):
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                    rewards[i][env.memory_num:]))

        # similarity after softmax
        similarity = readouts[0][0]["ValueMemory"]["similarity"].squeeze()
        plt.imshow(similarity, cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recall timestep")
        plt.title("recalling state-memory similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0], actions[0][env.memory_num:]))
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
        sim_mem[run_name_without_num].append(similarity)
        
        plt.imshow(similarity, cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recall timestep")
        plt.title("recalling state-memory similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0], actions[0][env.memory_num:]))
        savefig(fig_path, "similarity_memory")

        # similarity of states
        similarities = []
        for i in range(context_num):
            states = readouts[i][0]["state"].squeeze()
            similarity = skp.cosine_similarity(states, states)
            similarities.append(similarity)
        similarities = np.stack(similarities)
        similarity = np.mean(similarities, axis=0)

        plt.imshow(similarity[:timestep_each_phase, :timestep_each_phase], cmap="Blues")
        plt.colorbar()
        plt.title("encoding state similarity".format(memory_contexts[0], actions[0][env.memory_num:]))
        savefig(fig_path, "similarity_state_encode")
        sim_enc[run_name_without_num].append(similarity[:timestep_each_phase, :timestep_each_phase])

        plt.imshow(similarity[timestep_each_phase:, timestep_each_phase:], cmap="Blues")
        plt.colorbar()
        plt.title("recalling state similarity".format(memory_contexts[0], actions[0][env.memory_num:]))
        savefig(fig_path, "similarity_state_recall")
        sim_rec[run_name_without_num].append(similarity[timestep_each_phase:, timestep_each_phase:])

        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, :timestep_each_phase], cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recalling timestep")
        plt.title("encoding-recalling state similarity".format(memory_contexts[0], actions[0][env.memory_num:]))
        savefig(fig_path, "similarity_state_encode_recall")
        sim_enc_rec[run_name_without_num].append(similarity[timestep_each_phase:timestep_each_phase*2, :timestep_each_phase])

        # memory gate
        plt.figure(figsize=(4, 3), dpi=250)
        for i in range(context_num):
            readout = readouts[i][0]
            em_gates = readout['mem_gate_recall']
            plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[:env.memory_num], label="context {}".format(i))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("timesteps of recalling phase")
        plt.ylabel("memory gate")
        plt.tight_layout()
        savefig(fig_path, "em_gate_recall")

        # recall probability (output)
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, env.memory_num:])
        recall_probability.visualize(fig_path/"recall_prob")
        rec_prob[run_name_without_num].append(recall_probability.get_results())

        # recall probability (memory)
        retrieved_memories = []
        for i in range(context_num):
            retrieved_memory = readouts[i][0]["ValueMemory"]["retrieved_memory"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)
        retrieved_memories = np.stack(retrieved_memories)
        recall_probability = RecallProbability()
        recall_probability.fit(np.repeat(np.array(range(env.memory_num)).reshape(1, -1), context_num, axis=0), retrieved_memories)
        recall_probability.visualize(fig_path/"recall_prob_memory")
        rec_prob_mem[run_name_without_num].append(recall_probability.get_results())

        # PCA
        states = []
        for i in range(context_num):
            states.append(readouts[i][0]['state'])
        states = np.stack(states).squeeze()
        
        pca = PCA()
        pca.fit(states)
        pca.visualize_state_space(save_path=fig_path/"pca"/"memorizing", end_step=timestep_each_phase)
        pca.visualize_state_space(save_path=fig_path/"pca"/"recalling", start_step=timestep_each_phase, end_step=timestep_each_phase*2)
        pca.visualize_state_space(save_path=fig_path/"pca")

        half_states = []
        for i in range(context_num):
            half_states.append(readouts[i][0]['half_state'])
        half_states = np.stack(half_states).squeeze()
        pca = PCA()
        pca.fit(half_states)
        pca.visualize_state_space(save_path=fig_path/"pca"/"half_state"/"memorizing", end_step=timestep_each_phase)
        pca.visualize_state_space(save_path=fig_path/"pca"/"half_state"/"recalling", start_step=timestep_each_phase, end_step=timestep_each_phase*2)
        pca.visualize_state_space(save_path=fig_path/"pca"/"half_state")

        # SVM
        c_memorizing = np.stack([readouts[i][0]['state'][:timestep_each_phase].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)
        c_recalling = np.stack([readouts[i][0]['state'][-timestep_each_phase:].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)
        memory_sequence = np.stack([memory_contexts[i] for i in range(all_context_num)]).transpose(1, 0) - 1
        
        svm = SVM()
        svm.fit(c_memorizing, memory_sequence)
        svm.visualize(save_path=fig_path/"svm"/"c_mem")
        svm.visualize_by_memory(save_path=fig_path/"svm"/"c_mem")

        svm.fit(c_recalling, memory_sequence)
        svm.visualize(save_path=fig_path/"svm"/"c_rec")
        svm.visualize_by_memory(save_path=fig_path/"svm"/"c_rec")

    for run_name in run_names_without_num:
        plt.imshow(np.mean(np.stack(sim_mem[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recall timestep")
        plt.title("recalling state-memory similarity\nmean of {} models".format(run_num))
        savefig(paths['fig']/"mean"/run_name, "similarity_memory_mean")

        plt.imshow(np.mean(np.stack(sim_enc[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.title("encoding state similarity\nmean of {} models".format(run_num))
        savefig(paths['fig']/"mean"/run_name, "similarity_state_encode_mean")

        plt.imshow(np.mean(np.stack(sim_rec[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.title("recalling state similarity\nmean of {} models".format(run_num))
        savefig(paths['fig']/"mean"/run_name, "similarity_state_recall_mean")

        plt.imshow(np.mean(np.stack(sim_enc_rec[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recalling timestep")
        plt.title("encoding-recalling state similarity\nmean of {} models".format(run_num))
        savefig(paths['fig']/"mean"/run_name, "similarity_state_encode_recall_mean")

        recall_probability = RecallProbability()
        recall_probability.set_results(np.mean(np.stack(rec_prob[run_name]), axis=0))
        recall_probability.visualize(save_path=paths['fig']/"mean"/run_name/"recall_prob")

        recall_probability = RecallProbability()
        recall_probability.set_results(np.mean(np.stack(rec_prob_mem[run_name]), axis=0))
        recall_probability.visualize(save_path=paths['fig']/"mean"/run_name/"recall_prob_memory")


    # accuracy_list = []
    # for run_name in run_names_without_num:
    #     accuracy_list.append(np.mean(accuracy[run_name]))

    # for i in range(3):
    #     plt.plot([1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], accuracy_list[i*7:(i+1)*7], label="{} steps per item".format(i+1))
    # plt.legend()
    # plt.xlabel("a = dt/tau")
    # plt.ylabel("performance")
    # savefig(paths['fig']/"mean", "{}_performance".format(exp_name))
