import os
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.linear_model import RidgeClassifier

from utils import savefig
from analysis.behavior import RecallProbability, RecallProbabilityInTime
from analysis.decoding import ItemIdentityDecoder, DMAnswerDecoder, Classifier


def run(data_all, model_all, env, paths, exp_name):
    """
    separating trials with different number of matched items
    """

    plt.rcParams['font.size'] = 14

    run_num = len(list(data_all.keys()))

    env = env[0]

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        run_num = run_name.split("-")[1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)

        print()
        print(run_name)
        
        model = model_all[run_name]

        data = data[0]
        context_num = len(data["actions"])
        timestep_each_phase = env.unwrapped.sequence_len

        readouts = data['readouts']
        actions = data['actions']
        actions_len = []
        for i in actions:
            actions_len.append(len(i))
        actions_array = np.ones((context_num, max(actions_len))) * 2
        for i in range(context_num):
            actions_array[i, :actions_len[i]] = actions[i].reshape(-1)
        actions = actions_array
        actions = actions.reshape(-1, actions.shape[-1])


        """ compute accuracy """
        model_answers_all = []
        for i in range(context_num):
            answered = False
            for j in range(timestep_each_phase):
                if actions[i][timestep_each_phase+j] != 2:
                    answered = True
                    model_answers_all.append(actions[i][timestep_each_phase+j])
                    break
            if not answered:
                model_answers_all.append(actions[i][-1])
        model_answers_all = np.array(model_answers_all)

        print("model_answers_all: ", model_answers_all.shape)

        correct_num = 0
        for i in range(context_num):
            if model_answers_all[i] == data["trial_data"][i]["correct_answer"]:
                correct_num += 1
        accuracy = correct_num / context_num
        print("accuracy: ", accuracy)
        with open(fig_path/"accuracy.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([accuracy])



        """ gather data from all trials """
        matched_mask = []
        unmatched_mask = []

        sum_features = []
        c_memorizing = []
        c_recalling = []

        memory_sequences = []
        correct_trials = []
        valid_trials = []

        memory_similarities = []
        retrieved_memory_indexes = []
        retrieved_memories = []
        em_gates = []

        num_timesteps = np.zeros(timestep_each_phase)
        model_answers = []
        answer_timesteps = []
        actions_mask = []

        trials_by_matched_num = [[] for _ in range(timestep_each_phase+1)]
        correct_trials_by_matched_num = [[] for _ in range(timestep_each_phase+1)]

        # retrieved_memories = []
        # answers = []   
        # answers = model_answers
        for i in range(context_num):
            if data["trial_data"][i]["correct_answer"] == model_answers_all[i]:
                correct_trials.append(i)
            # print(readouts[i]["ValueMemory"]["similarity"].shape)
            retrieved_memory = np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(1), axis=-1)
            if len(retrieved_memory.shape) == 0:
                raise ValueError("trial {}, retrieved_memory shape is 0".format(i))
                # continue

            # sum feature (feature related to the question)
            sum_feature_index = data["trial_data"][i]["sum_feature"]
            memory_sequence = data["trial_data"][i]["memory_sequence"]
            sum_feature = memory_sequence[:, sum_feature_index]
            sum_features.append(sum_feature)

            # masks for matched and unmatched stimuli to the question
            matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
            unmatched_stimuli = [j for j in range(timestep_each_phase) if j not in matched_stimuli]
            matched_mask.append(np.array([1 if j in matched_stimuli else 0 for j in np.arange(timestep_each_phase)]))
            unmatched_mask.append(np.array([1 if j in unmatched_stimuli else 0 for j in np.arange(timestep_each_phase)]))

            trials_by_matched_num[len(matched_stimuli)].append(i)
            if data["trial_data"][i]["correct_answer"] == model_answers_all[i]:
                correct_trials_by_matched_num[len(matched_stimuli)].append(i)
            
            # states in encoding and recall phase
            c_memorizing.append(readouts[i]['state'][:timestep_each_phase].squeeze())
            c_rec = np.zeros((timestep_each_phase, readouts[i]['state'].shape[-1]))
            c_rec_len = readouts[i]['state'].shape[0] - timestep_each_phase - 1
            c_rec[:c_rec_len] = readouts[i]['state'][-c_rec_len:].squeeze()
            c_recalling.append(c_rec)

            # ground truthmemory sequence, items in integer
            memory_sequences.append(data["trial_data"][i]["memory_sequence_int"])

            # index of the most similar memory for each timestep
            retrieved_memory_index = np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(1), axis=-1)
            ms = np.ones((timestep_each_phase)) * -1
            ms[:retrieved_memory_index.shape[0]] = retrieved_memory_index
            retrieved_memory_indexes.append(ms)

            # memory similarity
            memory_similarity = np.zeros((timestep_each_phase, timestep_each_phase))
            memory_similarity_len = readouts[i]["ValueMemory"]["similarity"].shape[0]
            memory_similarity[:memory_similarity_len, :] = readouts[i]["ValueMemory"]["similarity"].squeeze()
            memory_similarities.append(memory_similarity)

            # retrieved memory, after possible weighted sum
            retrieved_memory = readouts[i]["retrieved_memory"].squeeze()
            rm = np.zeros((timestep_each_phase, retrieved_memory.shape[-1]))
            rm[:retrieved_memory.shape[0]] = retrieved_memory
            retrieved_memories.append(rm)

            # em gate
            em_gate = np.zeros((timestep_each_phase))
            em_gate_len = readouts[i]['mem_gate_recall'].shape[0]
            em_gate[:em_gate_len] = readouts[i]['mem_gate_recall'].squeeze()
            em_gates.append(em_gate)

            # compute number of timesteps taken to answer
            answered = False
            action_mask = np.ones(timestep_each_phase, dtype=bool)
            for j in range(timestep_each_phase):
                if actions[i][timestep_each_phase+j] != 2:
                    answered = True
                    num_timesteps[j] += 1
                    answer_timesteps.append(j+1)
                    model_answers.append(actions[i][timestep_each_phase+j])
                    if j != timestep_each_phase-1:
                        action_mask[j+1:] = False
                    break
            if not answered:
                model_answers.append(actions[i][-1])
                answer_timesteps.append(timestep_each_phase)
            actions_mask.append(action_mask)

            if i < 5:
                print("memory similarity sample:")
                print(readouts[i]["ValueMemory"]["similarity"].squeeze())

        correct_trials = np.array(correct_trials)
        correct_trials_num = correct_trials.shape[0]

        sum_features = np.stack(sum_features, axis=0)

        matched_mask = np.stack(matched_mask, axis=0)
        unmatched_mask = np.stack(unmatched_mask, axis=0)

        c_memorizing = np.stack(c_memorizing, axis=0)
        c_recalling = np.stack(c_recalling, axis=0)
        memory_sequences = np.stack(memory_sequences, axis=0)

        em_gates = np.stack(em_gates, axis=0)
        memory_similarities = np.stack(memory_similarities, axis=0)
        retrieved_memory_indexes = np.stack(retrieved_memory_indexes, axis=0)
        retrieved_memories = np.stack(retrieved_memories, axis=0)
        print(memory_similarities.shape, retrieved_memory_indexes.shape, retrieved_memories.shape)

        answers = np.array(model_answers)
        answer_timesteps = np.array(answer_timesteps)
        actions_mask = np.stack(actions_mask, axis=0)

        print("correct_trials_num: ", correct_trials_num)

        print("trials_by_matched_num: ", [len(trials_by_matched_num[i]) for i in range(timestep_each_phase+1)])
        print("correct_trials_by_matched_num: ", [len(correct_trials_by_matched_num[i]) for i in range(timestep_each_phase+1)])




        ################# analysis on all trials #################
        """ whether trials with more time steps have higher accuracy """
        accuracy_by_timestep_all = []
        plt.figure(figsize=(4.7, 3), dpi=180)
        for i in range(1, timestep_each_phase+1):
            accuracy_by_timestep = np.zeros(timestep_each_phase)
            trials_by_timestep = np.zeros(timestep_each_phase)
            for j in trials_by_matched_num[i]:
                if data["trial_data"][j]["correct_answer"] == model_answers_all[j]:
                    accuracy_by_timestep[answer_timesteps[j]-1] += 1
                trials_by_timestep[answer_timesteps[j]-1] += 1
            trials_by_timestep[trials_by_timestep == 0] = 1
            accuracy_by_timestep = accuracy_by_timestep / trials_by_timestep
            plt.plot(np.arange(1, timestep_each_phase+1), accuracy_by_timestep, label="{} matched".format(i), marker="o")
            accuracy_by_timestep_all.append(accuracy_by_timestep)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("time in recall phase")
        plt.ylabel("accuracy")
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        savefig(fig_path, "accuracy_timesteps")

        accuracy_by_timestep_all = np.array(accuracy_by_timestep_all)
        os.makedirs(fig_path/ "data", exist_ok=True)
        np.save(fig_path/ "data" / "accuracy_by_timestep_all.npy", accuracy_by_timestep_all)


        """ whether trials recalling more memories have higher accuracy """
        """ also: whether trials recalling more matched memories have higher accuracy """
        os.makedirs(fig_path/"retrieved_memory_num", exist_ok=True)

        num_retrieved_memories = []
        num_matched_memories_retrieved = []
        proportion_matched_memories_retrieved = []
        for i in range(context_num):
            mem_retrieved = np.zeros(timestep_each_phase)
            matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
            for j in range(timestep_each_phase):
                if int(retrieved_memory_indexes[i, j]) != -1:
                    mem_retrieved[int(retrieved_memory_indexes[i, j])] = 1
            num_retrieved_memories.append(np.sum(mem_retrieved))
            num_matched_memories_retrieved.append(np.sum(mem_retrieved[matched_stimuli]))
            total_matched_num = len(matched_stimuli)
            if total_matched_num == 0:
                total_matched_num = 1
            proportion_matched_memories_retrieved.append(np.sum(mem_retrieved[matched_stimuli]) / total_matched_num)
        num_retrieved_memories = np.array(num_retrieved_memories)
        num_matched_memories_retrieved = np.array(num_matched_memories_retrieved)
        proportion_matched_memories_retrieved = np.array(proportion_matched_memories_retrieved)

        # number of retrieved memories for trials with different number of timesteps
        num_retrieved_memories_mean_by_timestep = np.zeros((timestep_each_phase+1, timestep_each_phase))
        num_matched_memories_retrieved_mean_by_timestep = np.zeros((timestep_each_phase+1, timestep_each_phase))

        for j in range(timestep_each_phase+1):
            answer_timesteps_j = answer_timesteps[trials_by_matched_num[j]]
            for i in range(timestep_each_phase):
                num_retrieved_memories_mean_by_timestep[j, i] = np.mean(num_retrieved_memories[trials_by_matched_num[j]][answer_timesteps_j == i+1])
                num_matched_memories_retrieved_mean_by_timestep[j, i] = np.mean(num_matched_memories_retrieved[trials_by_matched_num[j]][answer_timesteps_j == i+1])

        plt.figure(figsize=(4.7, 3), dpi=180)
        for i in range(1, timestep_each_phase+1):
            plt.plot(np.arange(1, timestep_each_phase+1), num_retrieved_memories_mean_by_timestep[i], label="{} matched".format(i), marker="o")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("time in recall phase")
        plt.ylabel("number of\nretrieved memories")
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        savefig(fig_path/"retrieved_memory_num", "mean_by_timestep")
        np.save(fig_path/ "data" / "num_retrieved_memories_mean_by_timestep.npy", num_retrieved_memories_mean_by_timestep)

        plt.figure(figsize=(4.7, 3), dpi=180)
        for i in range(1, timestep_each_phase+1):
            plt.plot(np.arange(1, timestep_each_phase+1), num_matched_memories_retrieved_mean_by_timestep[i], label="{} matched".format(i), marker="o")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("time in recall phase")
        plt.ylabel("number of matched\nmemories retrieved")
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        savefig(fig_path/"retrieved_memory_num", "mean_by_timestep_matched")
        np.save(fig_path/ "data" / "num_matched_memories_retrieved_mean_by_timestep.npy", num_matched_memories_retrieved_mean_by_timestep)


        # accuracy for trials with different number of retrieved memories
        accuracy_by_num_retrieved_memories = np.zeros((timestep_each_phase+1, timestep_each_phase))
        num_trials_by_num_retrieved_memories = np.zeros((timestep_each_phase+1, timestep_each_phase))
        for i in range(1, timestep_each_phase+1):
            for j in trials_by_matched_num[i]:
                if data["trial_data"][j]["correct_answer"] == model_answers_all[j]:
                    accuracy_by_num_retrieved_memories[i, int(num_retrieved_memories[j]-1)] += 1
                num_trials_by_num_retrieved_memories[i, int(num_retrieved_memories[j]-1)] += 1
        num_trials_by_num_retrieved_memories[num_trials_by_num_retrieved_memories == 0] = 1
        accuracy_by_num_retrieved_memories = accuracy_by_num_retrieved_memories / num_trials_by_num_retrieved_memories

        plt.figure(figsize=(4.7, 3), dpi=180)
        for i in range(1, timestep_each_phase+1):
            plt.plot(np.arange(1, timestep_each_phase+1), accuracy_by_num_retrieved_memories[i], label="{} matched".format(i), marker="o")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("number of retrieved memories")
        plt.ylabel("accuracy")
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        savefig(fig_path/"retrieved_memory_num", "accuracy_by_num_retrieved_memories")
        np.save(fig_path/ "data" / "accuracy_by_num_retrieved_memories.npy", accuracy_by_num_retrieved_memories)


