import numpy as np
import matplotlib.pyplot as plt
import csv

import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge

from utils import savefig
from analysis.decomposition import PCA
from analysis.behavior import RecallProbability, RecallProbabilityInTime
from analysis.decoding import ItemIdentityDecoder, DMAnserDecoder


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

        # for i in range(10):
        #     retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
        #     print(retrieved_memory)
        #     print(np.argmax(retrieved_memory, axis=-1))
        #     matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
        #     print(matched_stimuli)
        #     print(data["trial_data"][i]["correct_answer"], actions[i][-1])
        #     print(data["trial_data"][i]["memory_sequence"])
        #     print(data["trial_data"][i]["question_feature"], 
        #           data["trial_data"][i]["question_value"],
        #           data["trial_data"][i]["sum_feature"])
        #     states = readouts[i]["state"].squeeze()
        #     # print(np.sum(np.abs(states), axis=-1))
        #     print()

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

            plt.figure(figsize=(4, 3), dpi=180)
            mean_em_gate = np.mean(em_gates, axis=0)
            std_em_gate = np.std(em_gates, axis=0)
            plt.plot(mean_em_gate, label="mean")
            plt.fill_between(np.arange(mean_em_gate.shape[0]), mean_em_gate - std_em_gate, mean_em_gate + std_em_gate, alpha=0.3)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("time of recall phase")
            plt.ylabel("memory gate")
            plt.tight_layout()
            savefig(fig_path, "em_gate_recall_mean")


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

        """ gather data from correct trials """
        sum_features = []
        matched_mask = []
        unmatched_mask = []
        c_memorizing = []
        c_recalling = []
        memory_sequences = []
        retrieved_memories = []     
        answers = []   
        for i in range(context_num):
            if data["trial_data"][i]["correct_answer"] != actions[i][-1]:
                continue
            sum_feature_index = data["trial_data"][i]["sum_feature"]
            memory_sequence = data["trial_data"][i]["memory_sequence"]
            sum_feature = memory_sequence[:, sum_feature_index]
            sum_features.append(sum_feature)

            matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
            unmatched_stimuli = [j for j in range(env.sequence_len) if j not in matched_stimuli]
            matched_mask.append(np.array([1 if j in matched_stimuli else 0 for j in np.arange(env.sequence_len)]))
            unmatched_mask.append(np.array([1 if j in unmatched_stimuli else 0 for j in np.arange(env.sequence_len)]))
            
            c_memorizing.append(readouts[i]['state'][:timestep_each_phase].squeeze())
            c_recalling.append(readouts[i]['state'][-timestep_each_phase:].squeeze())
            memory_sequences.append(data["trial_data"][i]["memory_sequence_int"])
            retrieved_memory = np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=-1)
            retrieved_memories.append(retrieved_memory)

            answers.append(actions[i][-1])

        sum_features = np.stack(sum_features, axis=0)
        matched_mask = np.stack(matched_mask, axis=0)
        unmatched_mask = np.stack(unmatched_mask, axis=0)
        c_memorizing = np.stack(c_memorizing, axis=0)
        c_recalling = np.stack(c_recalling, axis=0)
        memory_sequences = np.stack(memory_sequences, axis=0)
        retrieved_memories = np.stack(retrieved_memories, axis=0)
        # print(sum_features.shape, matched_mask.shape, unmatched_mask.shape)

        # c_memorizing = np.stack([readouts[i]['state'][:timestep_each_phase].squeeze() for i in range(context_num)])   # context_num * time * state_dim
        # c_recalling = np.stack([readouts[i]['state'][-timestep_each_phase:].squeeze() for i in range(context_num)])
        # print(c_memorizing.shape, c_recalling.shape)
        
        """ plot actions """
        num_actions = np.zeros((3, env.sequence_len))
        num_total_trials = 0
        for i in range(context_num):
            if data["trial_data"][i]["correct_answer"] != actions[i][-1]:
                continue
            for j in range(env.sequence_len):
                num_actions[actions[i][env.sequence_len+j], j] += 1
            num_total_trials += 1
        num_actions = num_actions / num_total_trials
        plt.figure(figsize=(4, 3), dpi=180)
        plt.plot(num_actions[0], label="action 0")
        plt.plot(num_actions[1], label="action 1")
        plt.plot(num_actions[2], label="no action")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("time in recall phase")
        plt.ylabel("proportion of actions")
        plt.legend()
        plt.tight_layout()
        savefig(fig_path, "actions")


        """ plot memories retrieved at each timestep """
        # print(retrieved_memories.shape, retrieved_memories[5:])
        recall_probability = RecallProbabilityInTime()
        recall_probability.fit(np.repeat(np.arange(4).reshape(1,-1),retrieved_memories.shape[0],axis=0), retrieved_memories)
        recall_probability.visualize(fig_path, timesteps=[0, 1, 2, 3])
        recall_probability.visualize_in_time(fig_path)


        """ decoding final answer """
        ridge_decoder = RidgeClassifier()
        decoder = DMAnserDecoder(decoder=ridge_decoder)
        decoder.fit(c_recalling.transpose(1, 0, 2), answers)
        decoder.visualize(save_path=fig_path, save_name="answer_decode", xlabel="time in recall phase", figsize=(4, 3.3))

        """ decoding """
        ridge_decoder = RidgeClassifier()
        ridge = ItemIdentityDecoder(decoder=ridge_decoder)

        ridge_encoding_item = ridge.fit(c_memorizing.transpose(1, 0, 2), memory_sequences.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_item", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_encoding_item.npy", ridge_encoding_item)

        ridge_encoding_sum_feature = ridge.fit(c_memorizing.transpose(1, 0, 2), sum_features.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_sumf", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_encoding_sum_feature.npy", ridge_encoding_sum_feature)

        ridge_encoding_sum_feature_matched = ridge.fit(c_memorizing.transpose(1, 0, 2), sum_features.transpose(1, 0), mask=matched_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_sumf_mat", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_encoding_sum_feature_matched.npy", ridge_encoding_sum_feature_matched)

        ridge_encoding_sum_feature_unmatched = ridge.fit(c_memorizing.transpose(1, 0, 2), sum_features.transpose(1, 0), mask=unmatched_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_sumf_unmat", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_encoding_sum_feature_unmatched.npy", ridge_encoding_sum_feature_unmatched)

        ridge_recall_item = ridge.fit(c_recalling.transpose(1, 0, 2), memory_sequences.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_item", colormap_label="item position\nin study order",
                                xlabel="time in recall phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_decoding_item.npy", ridge_recall_item)

        ridge_recall_sum_feature = ridge.fit(c_recalling.transpose(1, 0, 2), sum_features.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_sumf", colormap_label="item position\nin study order",
                                xlabel="time in recall phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_decoding_sum_feature.npy", ridge_recall_sum_feature)

        ridge_recall_sum_feature_matched = ridge.fit(c_recalling.transpose(1, 0, 2), sum_features.transpose(1, 0), mask=matched_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_sumf_mat", colormap_label="item position\nin study order",
                                xlabel="time in recall phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_decoding_sum_feature_matched.npy", ridge_recall_sum_feature_matched)

        ridge_recall_sum_feature_unmatched = ridge.fit(c_recalling.transpose(1, 0, 2), sum_features.transpose(1, 0), mask=unmatched_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_sumf_unmat", colormap_label="item position\nin study order",
                                xlabel="time in recall phase", figsize=(4, 3.3))
        np.save(fig_path/"ridge_decoding_sum_feature_unmatched.npy", ridge_recall_sum_feature_unmatched)


        # recall_probability = RecallProbability()
        # recall_probability.fit(memory_sequences, retrieved_memories)
        # # plot CRP curve
        # recall_probability.visualize_all_time(fig_path/"recall_prob")
        # recall_probability.visualize(fig_path/"recall_prob")
        # results_all_time = recall_probability.get_results_all_time()
        # # write to csv file
        # with open(fig_path/"recall_probability.csv", "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(results_all_time)

        # recall_probability_in_time = RecallProbabilityInTime()
        # recall_probability_in_time.fit(memory_sequences, retrieved_memories)
        # recall_probability_in_time.visualize(fig_path)
