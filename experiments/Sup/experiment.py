import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA, TDR
from analysis.decoding import SVM, PCSelectivity, SingleNeuronSelectivity
from analysis.behavior import RecallProbability, RecallProbabilityInTime
from analysis.dynamics import FixedPointAnalysis
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
    rec_prob_all = {}
    rec_prob_mem_all = {}
    rec_prob_by_time_all = {}
    tdr_variance = {}
    pc_selectivities = {}
    pc_explained_variances = {}

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
            rec_prob_all[run_name_without_num] = []
            rec_prob_mem_all[run_name_without_num] = []
            rec_prob_by_time_all[run_name_without_num] = []
            tdr_variance[run_name_without_num] = []
            pc_selectivities[run_name_without_num] = []
            pc_explained_variances[run_name_without_num] = []

    plt.rcParams['font.size'] = 14

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
        plt.xlabel("memories")
        plt.ylabel("recalling timestep")
        plt.title("memory similarity\nmean of 20 trials")
        savefig(fig_path, "recall_probability")
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

        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, timestep_each_phase:timestep_each_phase*2], cmap="Blues")
        plt.colorbar()
        plt.title("recalling state similarity".format(memory_contexts[0], actions[0][env.memory_num:]))
        savefig(fig_path, "similarity_state_recall")
        sim_rec[run_name_without_num].append(similarity[timestep_each_phase:timestep_each_phase*2, timestep_each_phase:timestep_each_phase*2])

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
        rec_prob_all[run_name_without_num].append(recall_probability.get_results_all_time())

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
        rec_prob_mem_all[run_name_without_num].append(recall_probability.get_results_all_time())

        # recall probability by time
        recall_probability_in_time = RecallProbabilityInTime()
        rec_prob_by_time = recall_probability_in_time.fit(memory_contexts, actions[:, env.memory_num:])
        rec_prob_by_time_all[run_name_without_num].append(rec_prob_by_time)

        # alignment of memory and output
        memory_output = []
        for i in range(context_num):
            memory_output.append(memory_contexts[i][retrieved_memories[i]])
        memory_output = np.stack(memory_output)
        alignment = np.mean(memory_output == actions[:context_num][:, env.memory_num:])
        print("alignment rate: {}".format(alignment))

        # PCA
        states = []
        for i in range(context_num):
            states.append(readouts[i][0]['state'])
        states = np.stack(states).squeeze()
        
        pca = PCA()
        pca.fit(states)
        pca.visualize_state_space(save_path=fig_path/"pca"/"memorizing", end_step=timestep_each_phase, title="encoding phase")
        pca.visualize_state_space(save_path=fig_path/"pca"/"recalling", start_step=timestep_each_phase, end_step=timestep_each_phase*2, title="recall phase")
        pca.visualize_state_space(save_path=fig_path/"pca")

        # fixed_point
        # fixed_point_analysis = FixedPointAnalysis()
        # fixed_point_analysis.fit(model, states, sample_num=20, time_step=20000)
        # fixed_point_analysis.visualize(save_path=fig_path/"fixed_point")

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

        # PC selectivity
        # convert actions and item index to one-hot
        actions_one_hot = np.zeros((all_context_num, env.memory_num, env.vocabulary_num))
        for i in range(all_context_num):
            actions_one_hot[i] = np.eye(env.vocabulary_num)[actions[i, env.memory_num:]-1]
        retrieved_memories = []
        for i in range(all_context_num):
            retrieved_memory = readouts[i][0]["ValueMemory"]["retrieved_memory"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)
        retrieved_memories = np.stack(retrieved_memories)
        # print(retrieved_memories.shape)
        retrieved_memories_one_hot = np.zeros((all_context_num, env.memory_num, env.memory_num))
        for i in range(all_context_num):
            retrieved_memories_one_hot[i] = np.eye(env.memory_num)[retrieved_memories[i]]
        labels = {"actions": actions_one_hot, "memory index": retrieved_memories_one_hot}
        # labels = {"actions": actions[:, env.memory_num:], "memory index": retrieved_memories}

        pc_selectivity = PCSelectivity(n_components=40)
        pc_selectivity.fit(c_memorizing, labels)
        pc_selectivity.visualize(save_path=fig_path/"pc_selectivity", file_name="memorizing")
        pc_s, pc_ev = pc_selectivity.fit(c_recalling, labels)
        pc_selectivity.visualize(save_path=fig_path/"pc_selectivity", file_name="recalling")
        pc_selectivities[run_name_without_num].append(pc_s)
        pc_explained_variances[run_name_without_num].append(pc_ev)

        # single neuron selectivity
        # single_neuron_selectivity = SingleNeuronSelectivity()
        # selectivity = single_neuron_selectivity.fit(c_memorizing, labels)
        # print("single neuron selectivity: {}".format(selectivity))

        # TDR
        states = []
        for i in range(context_num):
            states.append(readouts[i][0]['state'][:env.memory_num*2])
        states = np.stack(states).squeeze(2)
        tdr = TDR()
        # order_var_single_trial = np.arange(env.memory_num*2) % env.memory_num
        order_var_single_trial = np.arange(env.memory_num*2)
        order_var = np.repeat(order_var_single_trial.reshape(1, -1), context_num, axis=0)
        item_var = np.zeros((context_num, env.memory_num*2))
        for i in range(context_num):
            item_var[i, :env.memory_num] = memory_contexts[i]
            item_var[i, env.memory_num:env.memory_num*2] = actions[i][env.memory_num:]
        task_var = np.stack((order_var, item_var))
        task_var = np.transpose(task_var, (1, 2, 0))
        var_labels = ["index", "item"]
        tdr_variance[run_name_without_num].append(tdr.fit(states, task_var, var_labels))

    for run_name in run_names_without_num:
        plt.figure(figsize=(4, 3.5), dpi=180)
        plt.imshow(np.mean(np.stack(sim_mem[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.xlabel("memories")
        plt.ylabel("recalling timestep")
        plt.title("memory-recalling state similarity\nmean of {} models".format(run_num))
        plt.tight_layout()
        savefig(paths['fig']/"mean"/run_name, "state_memory_similarity_mean")

        plt.figure(figsize=(4, 3.5), dpi=180)
        plt.imshow(np.mean(np.stack(sim_enc[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.title("encoding state similarity\nmean of {} models".format(run_num))
        plt.tight_layout()
        savefig(paths['fig']/"mean"/run_name, "similarity_state_encode_mean")

        plt.figure(figsize=(4, 3.5), dpi=180)
        plt.imshow(np.mean(np.stack(sim_rec[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.title("recalling state similarity\nmean of {} models".format(run_num))
        plt.tight_layout()
        savefig(paths['fig']/"mean"/run_name, "similarity_state_recall_mean")

        plt.figure(figsize=(4, 3.5), dpi=180)
        plt.imshow(np.mean(np.stack(sim_enc_rec[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recalling timestep")
        plt.title("encoding-recalling state similarity\nmean of {} models".format(run_num))
        plt.tight_layout()
        savefig(paths['fig']/"mean"/run_name, "similarity_state_encode_recall_mean")

        plt.figure(figsize=(4.5, 4), dpi=180)
        plt.imshow(np.mean(np.stack(rec_prob_by_time_all[run_name]), axis=0), cmap="Blues")
        plt.colorbar()
        plt.xlabel("item position")
        plt.ylabel("recalling timestep")
        plt.title("output probability\nmean of {} models".format(run_num))
        plt.tight_layout()
        savefig(paths['fig']/"mean"/run_name, "recall_probability")

        recall_probability = RecallProbability()
        recall_probability.set_results(np.mean(np.stack(rec_prob[run_name]), axis=0), np.mean(np.stack(rec_prob_all[run_name]), axis=0))
        recall_probability.visualize(save_path=paths['fig']/"mean"/run_name/"recall_prob", title="output probability")

        recall_probability = RecallProbability()
        recall_probability.set_results(np.mean(np.stack(rec_prob_mem[run_name]), axis=0), np.mean(np.stack(rec_prob_mem_all[run_name]), axis=0))
        recall_probability.visualize(save_path=paths['fig']/"mean"/run_name/"recall_prob_memory", title="recall probability")

        tdr_variance_mean = np.mean(np.stack(tdr_variance[run_name]), axis=0)
        tdr_variance_std = np.std(np.stack(tdr_variance[run_name]), axis=0)
        plt.figure(figsize=(3, 3.5), dpi=180)
        plt.bar(np.arange(2), tdr_variance_mean, yerr=tdr_variance_std)
        plt.xticks(np.arange(2), ["index", "item"])
        plt.ylabel("explained variance")
        # plt.title("Variance of task parameters\nmean of {} models".format(run_num))
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        savefig(paths['fig']/"mean"/run_name, "tdr_variance")

        pc_selectivity = PCSelectivity(40)
        pc_selectivity.set_results(["item information", "temporal information"], 
                                    np.mean(np.stack(pc_selectivities[run_name]), axis=0),
                                    np.mean(np.stack(pc_explained_variances[run_name]), axis=0),
                                    np.std(np.stack(pc_selectivities[run_name]), axis=0),
                                    np.std(np.stack(pc_explained_variances[run_name]), axis=0))
        pc_selectivity.visualize(save_path=paths['fig']/"mean"/run_name)

    # accuracy_list = []
    # for run_name in run_names_without_num:
    #     accuracy_list.append(np.mean(accuracy[run_name]))

    # for i in range(3):
    #     plt.plot([1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], accuracy_list[i*7:(i+1)*7], label="{} steps per item".format(i+1))
    # plt.legend()
    # plt.xlabel("a = dt/tau")
    # plt.ylabel("performance")
    # savefig(paths['fig']/"mean", "{}_performance".format(exp_name))
