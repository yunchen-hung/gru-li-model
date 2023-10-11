import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import SVM
from analysis.behavior import RecallProbability, RecallProbabilityInTime
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

        # get recorded data and outputs of the model
        readouts = data['readouts']
        actions = data['actions']
        probs = data['probs']
        rewards = data['rewards']
        values = data['values']
        accuracy[run_name_without_num].append(data['accuracy'])

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)

        # convert data to numpy array
        memory_contexts = np.array(data['memory_contexts'])     # ground truth of memory for each trial
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        actions = np.array(actions)
        actions = actions.reshape(-1, actions.shape[-1])        # (trials, timesteps per trial)
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])        # (trials, timesteps per trial)

        if "ValueMemory" in readouts[0][0] and "similarity" in readouts[0][0]["ValueMemory"]:
            has_memory = True
        else:
            has_memory = False
        
        # print ground truths, actions and rewards for 5 trials
        for i in range(5):
            if has_memory:
                print("context {}, gt: {}, action: {}, retrieved memory: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                np.argmax(readouts[i][0]["ValueMemory"]["similarity"].squeeze(), axis=1)+1, rewards[i][env.memory_num:]))
            else:
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                rewards[i][env.memory_num:]))

        if has_memory:
            # similarity used for memory retrieval after softmax
            # one trial
            similarity = readouts[0][0]["ValueMemory"]["similarity"].squeeze()
            plt.imshow(similarity, cmap="Blues")
            plt.colorbar()
            plt.xlabel("encoding timestep")
            plt.ylabel("recall timestep")
            plt.title("recalling state-memory similarity\nmemory sequence {}\nrecall sequence {}".format(memory_contexts[0], actions[0][env.memory_num:]))
            savefig(fig_path, "recall_probability_one_trial")

            # average over all trials
            similarities = []
            for i in range(all_context_num):
                similarity = readouts[i][0]["ValueMemory"]["similarity"].squeeze()
                similarities.append(similarity)
            similarities = np.stack(similarities)
            similarity = np.mean(similarities, axis=0)
            plt.imshow(similarity, cmap="Blues")
            plt.colorbar()
            plt.xlabel("memories")
            plt.ylabel("recalling timestep")
            plt.title("memory similarity\nmean of {} trials".format(all_context_num))
            savefig(fig_path, "recall_probability")
            sim_mem[run_name_without_num].append(similarity)

            # average over all trials and normalize in each timestep
            similarity_normalized = similarity / np.sum(similarity, axis=1)
            plt.imshow(similarity_normalized, cmap="Blues")
            plt.colorbar()
            plt.xlabel("memories")
            plt.ylabel("recalling timestep")
            plt.title("memory similarity\nmean of {} trials\nnormalized in each timestep".format(all_context_num))
            savefig(fig_path, "recall_probability_normalized")

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
        plt.title("encoding state similarity")
        savefig(fig_path, "similarity_state_encode")
        sim_enc[run_name_without_num].append(similarity[:timestep_each_phase, :timestep_each_phase])

        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, timestep_each_phase:timestep_each_phase*2], cmap="Blues")
        plt.colorbar()
        plt.title("recalling state similarity")
        savefig(fig_path, "similarity_state_recall")
        sim_rec[run_name_without_num].append(similarity[timestep_each_phase:timestep_each_phase*2, timestep_each_phase:timestep_each_phase*2])

        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, :timestep_each_phase], cmap="Blues")
        plt.colorbar()
        plt.xlabel("encoding timestep")
        plt.ylabel("recalling timestep")
        plt.title("encoding-recalling state similarity")
        savefig(fig_path, "similarity_state_encode_recall")
        sim_enc_rec[run_name_without_num].append(similarity[timestep_each_phase:timestep_each_phase*2, :timestep_each_phase])

        # memory gate
        if "mem_gate_recall" in readouts[0][0]:
            plt.figure(figsize=(4, 3), dpi=250)
            for i in range(context_num):
                em_gates = readouts[i][0]['mem_gate_recall']
                plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[:timestep_each_phase], label="context {}".format(i))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("timesteps of recalling phase")
            plt.ylabel("memory gate")
            plt.tight_layout()
            savefig(fig_path, "em_gate_recall")

        # recall probability (output)
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        recall_probability.visualize(fig_path/"recall_prob")
        rec_prob[run_name_without_num].append(recall_probability.get_results())
        rec_prob_all[run_name_without_num].append(recall_probability.get_results_all_time())

        # recall probability by time
        recall_probability_in_time = RecallProbabilityInTime()
        rec_prob_by_time = recall_probability_in_time.fit(memory_contexts, actions[:, -timestep_each_phase:])
        recall_probability_in_time.visualize(fig_path)
        rec_prob_by_time_all[run_name_without_num].append(rec_prob_by_time)

        # recall probability (memory)
        if has_memory:
            retrieved_memories = []
            for i in range(all_context_num):
                retrieved_memory = readouts[i][0]["ValueMemory"]["retrieved_memory"].squeeze()
                retrieved_memory = np.argmax(retrieved_memory, axis=-1)
                retrieved_memories.append(retrieved_memory)
            retrieved_memories = np.stack(retrieved_memories)
            recall_probability = RecallProbability()
            recall_probability.fit(np.repeat(np.array(range(env.memory_num)).reshape(1, -1), context_num, axis=0), retrieved_memories)
            recall_probability.visualize(fig_path/"recall_prob_memory")
            rec_prob_mem[run_name_without_num].append(recall_probability.get_results())
            rec_prob_mem_all[run_name_without_num].append(recall_probability.get_results_all_time())

            # alignment of memory and output
            memory_output = []
            for i in range(all_context_num):
                memory_output.append(memory_contexts[i][retrieved_memories[i]])
            memory_output = np.stack(memory_output)
            alignment = np.mean(memory_output == actions[:all_context_num][:, env.memory_num:])
            print("alignment rate: {}".format(alignment))

            # matrix of alignment of memory and output
            # the probability of "when the output is 1~n, the probability of retrieving memory 1~n"
            memory_index = retrieved_memories + 1
            output_index = []
            for i in range(all_context_num):
                index = []
                for t in range(env.memory_num):
                    position1 = np.where(memory_contexts[i] == actions[i][t])
                    if position1[0].shape[0] != 0:
                        position1 = position1[0][0] + 1
                        index.append(position1)
                    else:
                        index.append(0)
                output_index.append(index)
            output_index = np.stack(output_index)
            alignment_matrix = np.zeros((env.memory_num, env.memory_num))
            for i in range(all_context_num):
                for t in range(env.memory_num):
                    if output_index[i][t] != 0:
                        alignment_matrix[output_index[i][t]-1][memory_index[i][t]-1] += 1
            alignment_matrix = alignment_matrix / np.sum(alignment_matrix, axis=1, keepdims=True)
            plt.imshow(alignment_matrix, cmap="Blues")
            plt.colorbar()
            plt.xlabel("memory")
            plt.ylabel("output")
            plt.title("probability of retrieving each memory")
            savefig(fig_path, "alignment_matrix")

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

        # svm.fit(c_recalling, memory_sequence)
        svm.fit(c_recalling, actions[:, -timestep_each_phase:].transpose(1, 0))
        svm.visualize(save_path=fig_path/"svm"/"c_rec")
        svm.visualize_by_memory(save_path=fig_path/"svm"/"c_rec")

    for run_name in run_names_without_num:
        if has_memory:
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

        if has_memory:
            recall_probability = RecallProbability()
            recall_probability.set_results(np.mean(np.stack(rec_prob_mem[run_name]), axis=0), np.mean(np.stack(rec_prob_mem_all[run_name]), axis=0))
            recall_probability.visualize(save_path=paths['fig']/"mean"/run_name/"recall_prob_memory", title="recall probability")
