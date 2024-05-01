import numpy as np
import matplotlib.pyplot as plt
import csv

from utils import savefig
from analysis.decomposition import PCA
from analysis.behavior import RecallProbability, RecallProbabilityInTime
import sklearn.metrics.pairwise as skp


def run(data_all, model_all, env, paths, exp_name):
    run_num = len(list(data_all.keys()))

    group1 = []
    group2 = []
    group3 = []

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        run_num = run_name.split("-")[1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)

        print()
        print(run_name)
        
        model = model_all[run_name]

        context_num = len(data["actions"])
        timestep_each_phase = env.sequence_len

        readouts = data['readouts']
        actions = data['actions']
        actions = np.array(actions).squeeze(-1)
        actions = actions.reshape(-1, actions.shape[-1])

        for i in range(10):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            # print(retrieved_memory)
            print(np.argmax(retrieved_memory, axis=-1))
            matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
            print(matched_stimuli)
            print(data["trial_data"][i]["correct_answer"], actions[i][-1])
            print(data["trial_data"][i]["memory_sequence"])
            print(data["trial_data"][i]["question_feature"], 
                  data["trial_data"][i]["question_value"],
                  data["trial_data"][i]["sum_feature"])
            states = readouts[i]["state"].squeeze()
            # print(np.sum(np.abs(states), axis=-1))
            print()

        em_gates = []
        if "mem_gate_recall" in readouts[0]:
            plt.figure(figsize=(4, 3), dpi=180)
            for i in range(context_num):
                em_gate = readouts[i]['mem_gate_recall']
                plt.plot(np.mean(em_gate.squeeze(1), axis=-1)[:timestep_each_phase], label="context {}".format(i))
                em_gates.append(em_gate.squeeze(1))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("time of recall phase")
            plt.ylabel("memory gate")
            plt.tight_layout()
            savefig(fig_path, "em_gate_recall")
        em_gates = np.stack(em_gates, axis=0).squeeze()

        retrieved_memories = []
        for i in range(context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memories.append(retrieved_memory)
        retrieved_memories = np.stack(retrieved_memories, axis=0)
        # print(np.argmax(retrieved_memories, axis=2)[:10])

        """ count the proportion of memories retrieved for items matched and unmatched with the query """
        answer_memory = 0.0
        nonanswer_memory = 0.0
        answer_memory_after_gate = 0.0
        nonanswer_memory_after_gate = 0.0
        for i in range(context_num):
            matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
            unmatched_stimuli = [j for j in range(env.sequence_len) if j not in matched_stimuli]
            # print(matched_stimuli)
            # print(retrieved_memories[i, :2, :])
            # print(retrieved_memories[i, :2, matched_stimuli])
            # print(retrieved_memories[i, :2, unmatched_stimuli])
            if len(matched_stimuli) == 0 or len(unmatched_stimuli) == 0:
                continue
            answer_memory += np.sum(retrieved_memories[i, :, matched_stimuli]) / len(matched_stimuli)
            nonanswer_memory += np.sum(retrieved_memories[i, :, unmatched_stimuli]) / len(unmatched_stimuli)
            answer_memory_after_gate += np.sum(retrieved_memories[i, :, matched_stimuli] * em_gates[i]) / len(matched_stimuli)
            nonanswer_memory_after_gate += np.sum(retrieved_memories[i, :, unmatched_stimuli] * em_gates[i]) / len(unmatched_stimuli)
        print("answer_memory: ", answer_memory)
        print("nonanswer_memory: ", nonanswer_memory)
        print("answer_memory_after_gate: ", answer_memory_after_gate)
        print("nonanswer_memory_after_gate: ", nonanswer_memory_after_gate)
