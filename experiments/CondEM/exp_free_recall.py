import csv
import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import SVM, PCSelectivity
from analysis.behavior import RecallProbability, RecallProbabilityInTime, TemporalFactor
import sklearn.metrics.pairwise as skp


def run(data_all, model_all, env, paths, exp_name):
    plt.rcParams['font.size'] = 14

    for run_name, data in data_all.items():
        fig_path = paths["fig"]/run_name
        fig_path.mkdir(parents=True, exist_ok=True)
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
        rewards = data['rewards']

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)

        # convert data to numpy array
        memory_contexts = np.array(trial_data['memory_sequence_int'] for trial_data in data['trial_data'])     # ground truth of memory for each trial
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        actions = np.array(actions)
        actions = actions.reshape(-1, actions.shape[-1])        # (trials, timesteps per trial)
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])        # (trials, timesteps per trial)

        if "ValueMemory" in readouts[0] and "similarity" in readouts[0]["ValueMemory"]:
            has_memory = True
        else:
            has_memory = False
        
        # print ground truths, actions and rewards for 5 trials
        for i in range(5):
            if has_memory:
                print("context {}, gt: {}, action: {}, retrieved memory: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=1)+1, rewards[i][env.memory_num:]))
            else:
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                rewards[i][env.memory_num:]))

        """ count temporal factor and forward asymmetry """
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        rec_prob_all_data = recall_probability.get_results_all_time()
        forward_asymmetry = rec_prob_all_data[timestep_each_phase] - rec_prob_all_data[timestep_each_phase-2]
        temporal_factor = TemporalFactor()
        temp_fact = temporal_factor.fit(memory_contexts, actions[:, -timestep_each_phase:])
        temp_fact = np.mean(temp_fact)
        print("forward asymmetry:[{},{}]".format(data['accuracy'], forward_asymmetry))
        print("temporal factor:[{},{}]".format(data['accuracy'], temp_fact))
        # write to csv file
        with open(fig_path/"contiguity_effect.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([data['accuracy'], forward_asymmetry, temp_fact])

        """ similarity of states """
        similarities = []
        for i in range(context_num):
            states = readouts[i]["state"].squeeze()
            similarity = skp.cosine_similarity(states, states)
            similarities.append(similarity)
        similarities = np.stack(similarities)
        similarity = np.mean(similarities, axis=0)

        plt.figure(figsize=(5, 4.2), dpi=180)
        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, :timestep_each_phase], cmap="Blues")
        plt.colorbar(label="cosine similarity")
        plt.xlabel("hidden states in encoding phase")
        plt.ylabel("hidden states in recalling phase")
        # plt.title("encoding-recalling state similarity")
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "encode_recall", format="svg")

        """ recall probability (output) (CRP curve) """
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        recall_probability.visualize_all_time(fig_path/"recall_prob", format="svg")
        results_all_time = recall_probability.get_results_all_time()
        # write to csv file
        with open(fig_path/"recall_probability.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(results_all_time)

        """ recall probability by time (recall probability matrix) """
        recall_probability_in_time = RecallProbabilityInTime()
        recall_probability_in_time.fit(memory_contexts, actions[:, -timestep_each_phase:])
        recall_probability_in_time.visualize(fig_path, format="svg")

        """ PCA """
        states = []
        for i in range(context_num):
            states.append(readouts[i]['state'])
        states = np.stack(states).squeeze()
        
        pca = PCA()
        pca.fit(states)
        pca.visualize_state_space(save_path=fig_path/"pca_state_space", end_step=timestep_each_phase, title="encoding phase", file_name="encoding", format="svg")
        pca.visualize_state_space(save_path=fig_path/"pca_state_space", start_step=timestep_each_phase, end_step=timestep_each_phase*2, title="recall phase",
                                file_name="recall", format="svg")

        """ SVM """
        c_memorizing = np.stack([readouts[i]['state'][:timestep_each_phase].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)   # time * context_num * state_dim
        c_recalling = np.stack([readouts[i]['state'][-timestep_each_phase:].squeeze() for i in range(all_context_num)]).transpose(1, 0, 2)
        memory_sequence = np.stack([memory_contexts[i] for i in range(all_context_num)]).transpose(1, 0) - 1
        
        svm = SVM()
        svm.fit(c_memorizing, memory_sequence)
        svm.visualize_by_memory(save_path=fig_path/"svm", save_name="c_mem", title="encoding phase", colormap_label="item in study order", format="svg")

        # svm.fit(c_recalling, memory_sequence)
        svm_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(env.memory_num):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    svm_mask[i][t] = 1
        svm.fit(c_recalling, actions[:, -timestep_each_phase:].transpose(1, 0), svm_mask.transpose(1, 0))
        svm.visualize_by_memory(save_path=fig_path/"svm", save_name="c_rec", title="recall phase", colormap_label="item in recall order", format="svg")

        """ PC selectivity """
        # convert actions and item index to one-hot
        actions_one_hot = np.zeros((all_context_num, env.memory_num, env.vocabulary_num))
        for i in range(all_context_num):
            actions_one_hot[i] = np.eye(env.vocabulary_num)[actions[i, env.memory_num:]-1]

        # memory content
        memories_one_hot = np.zeros((all_context_num, env.memory_num, env.vocabulary_num))
        for i in range(all_context_num):
            memories_one_hot[i] = np.eye(env.vocabulary_num)[memory_contexts[i]-1]

        # memory index
        retrieved_memories = []
        for i in range(all_context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)
        retrieved_memories = np.stack(retrieved_memories)
        # print(retrieved_memories.shape)
        retrieved_memories_one_hot = np.zeros((all_context_num, env.memory_num, env.memory_num))
        for i in range(all_context_num):
            retrieved_memories_one_hot[i] = np.eye(env.memory_num)[retrieved_memories[i]]
        
        # labels = {"actions": actions[:, env.memory_num:], "memory index": retrieved_memories}

        pc_selectivity = PCSelectivity(n_components=128)
        labels = {"memory content": memories_one_hot, "memory index": retrieved_memories_one_hot}
        pc_selectivity.fit(c_memorizing, labels)
        pc_selectivity.visualize(save_path=fig_path/"pc_selectivity", file_name="encoding")

        pc_selectivity = PCSelectivity(n_components=128)
        labels = {"recalled memory": actions_one_hot, "memory index": retrieved_memories_one_hot}
        pc_selectivity.fit(c_recalling, labels)
        pc_selectivity.visualize(save_path=fig_path/"pc_selectivity", file_name="recalling")
