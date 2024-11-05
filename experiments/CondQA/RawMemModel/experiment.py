import os
import numpy as np
import matplotlib.pyplot as plt
import csv

import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge

from utils import savefig
from analysis.decomposition import PCA
from analysis.behavior import RecallProbability, RecallProbabilityInTime
from analysis.decoding import ItemIdentityDecoder, DMAnswerDecoder, Classifier


def run(data_all, model_all, env, paths, exp_name):
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



        """ print sample trials """
        # for i in range(5):
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

        # print("example trials:")
        # for i in range(10):
        #     print(actions[i])
        #     print(data["trial_data"][i]["correct_answer"])
        # print()


        """ distribution of number of timesteps taken in recall phase """
        num_timesteps = np.zeros(timestep_each_phase)
        model_answers = []
        answer_timesteps = []       # the number of timesteps taken to answer for each trial
        actions_mask = np.ones((context_num, timestep_each_phase), dtype=bool)
        for i in range(context_num):
            answered = False
            for j in range(timestep_each_phase):
                if actions[i][timestep_each_phase+j] != 2:
                    answered = True
                    num_timesteps[j] += 1
                    answer_timesteps.append(j+1)
                    model_answers.append(actions[i][timestep_each_phase+j])
                    if j != timestep_each_phase-1:
                        actions_mask[i, j+1:] = False
                    break
            if not answered:
                model_answers.append(actions[i][-1])
                # answer_timesteps[j] += 1
                answer_timesteps.append(timestep_each_phase)
        answer_timesteps = np.array(answer_timesteps)
        prop_timesteps = num_timesteps / context_num
        plt.figure(figsize=(4, 3), dpi=180)
        plt.bar(np.arange(1, timestep_each_phase+1), prop_timesteps)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("time in recall phase")
        plt.ylabel("proportion of trials\nanswered")
        plt.tight_layout()
        savefig(fig_path, "answer_timesteps")



        """ compute accuracy """
        correct_num = 0
        for i in range(context_num):
            if model_answers[i] == data["trial_data"][i]["correct_answer"]:
                correct_num += 1
        accuracy = correct_num / context_num
        print("accuracy: ", accuracy)
        with open(fig_path/"accuracy.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([accuracy])
        


        """ em gate at each timestep """
        em_gates = []
        if "mem_gate_recall" in readouts[0]:
            for i in range(context_num):
                em_gate = np.zeros((timestep_each_phase))
                em_gate_len = readouts[i]['mem_gate_recall'].shape[0]
                em_gate[:em_gate_len] = readouts[i]['mem_gate_recall'].squeeze()
                em_gates.append(em_gate)
            em_gates = np.stack(em_gates, axis=0).squeeze()


        retrieved_memories = np.zeros((context_num, timestep_each_phase, timestep_each_phase))
        for i in range(context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memories[i, :retrieved_memory.shape[0], :] = retrieved_memory
        # print("retrieved_memories:", retrieved_memories.shape)
        # print(np.argmax(retrieved_memories, axis=2)[:10])


        """ gather data from correct trials """
        sum_features = []
        matched_mask = []
        unmatched_mask = []
        c_memorizing = []
        c_recalling = []
        memory_sequences = []
        correct_trials = []
        retrieved_memories = []
        # retrieved_memories = []
        # answers = []   
        # answers = model_answers
        for i in range(context_num):
            if data["trial_data"][i]["correct_answer"] != model_answers[i]:
                continue
            retrieved_memory = np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=-1)
            if len(retrieved_memory.shape) == 0:
                continue

            correct_trials.append(i)

            sum_feature_index = data["trial_data"][i]["sum_feature"]
            memory_sequence = data["trial_data"][i]["memory_sequence"]
            sum_feature = memory_sequence[:, sum_feature_index]
            sum_features.append(sum_feature)

            matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
            unmatched_stimuli = [j for j in range(timestep_each_phase) if j not in matched_stimuli]
            matched_mask.append(np.array([1 if j in matched_stimuli else 0 for j in np.arange(timestep_each_phase)]))
            unmatched_mask.append(np.array([1 if j in unmatched_stimuli else 0 for j in np.arange(timestep_each_phase)]))
            
            c_memorizing.append(readouts[i]['state'][:timestep_each_phase].squeeze())
            # c_recalling.append(readouts[i]['state'][-timestep_each_phase:].squeeze())
            c_rec = np.zeros((timestep_each_phase, readouts[i]['state'].shape[-1]))
            c_rec_len = readouts[i]['state'].shape[0] - timestep_each_phase - 1
            c_rec[:c_rec_len] = readouts[i]['state'][-c_rec_len:].squeeze()
            c_recalling.append(c_rec)

            memory_sequences.append(data["trial_data"][i]["memory_sequence_int"])

            retrieved_memory = np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=-1)
            rm = np.ones((timestep_each_phase)) * -1
            rm[:retrieved_memory.shape[0]] = retrieved_memory
            retrieved_memories.append(rm)

            # answers.append(actions[i][-1])

        correct_trials = np.array(correct_trials)
        sum_features = np.stack(sum_features, axis=0)
        matched_mask = np.stack(matched_mask, axis=0)
        unmatched_mask = np.stack(unmatched_mask, axis=0)
        c_memorizing = np.stack(c_memorizing, axis=0)
        c_recalling = np.stack(c_recalling, axis=0)
        memory_sequences = np.stack(memory_sequences, axis=0)
        answers = np.array(model_answers)[correct_trials]
        answer_timesteps = answer_timesteps[correct_trials]

        retrieved_memories = np.stack(retrieved_memories, axis=0)



        """ count the proportion of memories retrieved for items matched and unmatched with the query """
        # matched_stimuli_num = 0.0
        # unmatched_stimuli_num = 0.0
        # answer_memory = 0.0
        # nonanswer_memory = 0.0
        # answer_memory_after_gate = 0.0
        # nonanswer_memory_after_gate = 0.0
        # for i in range(correct_trials.shape[0]):
        #     matched_stimuli = data["trial_data"][i]["matched_stimuli_index"]
        #     unmatched_stimuli = [j for j in range(timestep_each_phase) if j not in matched_stimuli]
        #     matched_stimuli_num += len(matched_stimuli)
        #     unmatched_stimuli_num += len(unmatched_stimuli)
        #     if len(matched_stimuli) == 0 or len(unmatched_stimuli) == 0:
        #         continue
        #     answer_memory += np.sum(retrieved_memories[i, :, matched_stimuli] * actions_mask[i, :]) # / len(matched_stimuli)
        #     nonanswer_memory += np.sum(retrieved_memories[i, :, unmatched_stimuli] * actions_mask[i, :]) # / len(unmatched_stimuli)
        #     answer_memory_after_gate += np.sum(retrieved_memories[i, :, matched_stimuli] * actions_mask[i, :] * em_gates[i]) # / len(matched_stimuli)
        #     nonanswer_memory_after_gate += np.sum(retrieved_memories[i, :, unmatched_stimuli] * actions_mask[i, :] * em_gates[i]) # / len(unmatched_stimuli)
        # print("answer_memory: ", answer_memory)
        # print("nonanswer_memory: ", nonanswer_memory)
        # print("answer_memory_after_gate: ", answer_memory_after_gate)
        # print("nonanswer_memory_after_gate: ", nonanswer_memory_after_gate)
        # print("matched_stimuli_num: ", matched_stimuli_num)
        # print("unmatched_stimuli_num: ", unmatched_stimuli_num)
        # print("ratio:", np.round(answer_memory/matched_stimuli_num/nonanswer_memory*unmatched_stimuli_num, 4))
        # print("ratio after gate:", np.round(answer_memory_after_gate/matched_stimuli_num/nonanswer_memory_after_gate*unmatched_stimuli_num, 4))



        """ plot em gate """
        def plot_em_gate(trials, fig_name):
            plt.figure(figsize=(4, 3), dpi=180)
            for i in trials:
                plt.plot(em_gates[i][actions_mask[i]], label="context {}".format(i))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("time of recall phase")
            plt.ylabel("memory gate")
            plt.tight_layout()
            savefig(fig_path/"em_gate", fig_name)

        def plot_em_gate_mean(trials, fig_name):
            mask = actions_mask[trials]
            gate = em_gates[trials]
            mask_sum = np.sum(mask, axis=0)
            mask_sum[mask_sum==0] = 1
            mean_gate = np.sum(gate*mask, axis=0) / mask_sum
            std_gate = np.sqrt(np.sum((gate - mean_gate)**2*mask, axis=0) / mask_sum)

            plt.figure(figsize=(4, 3), dpi=180)
            plt.plot(mean_gate, label="mean")
            plt.fill_between(np.arange(mean_gate.shape[0]), mean_gate - std_gate, mean_gate + std_gate, alpha=0.3)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("time of recall phase")
            plt.ylabel("memory gate")
            plt.tight_layout()
            savefig(fig_path/"em_gate", fig_name)

        plot_em_gate(correct_trials, "all")
        plot_em_gate_mean(correct_trials, "mean_all")

        for j in range(1, timestep_each_phase+1):
            chosen_trials = np.where(answer_timesteps == j)[0]
            # print("chosen trials for timestep {}".format(j), chosen_trials)
            plot_em_gate(chosen_trials, "timestep_{}".format(j))
            plot_em_gate_mean(chosen_trials, "mean_timestep_{}".format(j))


        
        """ plot actions """
        def plot_actions(trials, timesteps, fig_name):
            num_actions = np.zeros((3, timestep_each_phase))
            num_total_trials = np.zeros(timestep_each_phase)
            for i in trials:
                # if data["trial_data"][i]["correct_answer"] != model_answers[i]:
                #     continue
                # retrieved_memory = np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=-1)
                # if len(retrieved_memory.shape) == 0:
                #     continue
                for j in range(timestep_each_phase):
                    if not actions_mask[i, j]:
                        continue
                    num_actions[int(actions[i][timestep_each_phase+j]), j] += 1
                    num_total_trials[j] += 1
            num_total_trials[num_total_trials==0] = 1
            num_actions = num_actions / num_total_trials
            plt.figure(figsize=(4, 3), dpi=180)
            if timesteps == 1:
                plt.scatter([1], num_actions[0][:timesteps], label="action 0")
                plt.scatter([1], num_actions[1][:timesteps], label="action 1")
                plt.scatter([1], num_actions[2][:timesteps], label="no action")
            else:
                plt.plot(np.arange(1, timesteps+1), num_actions[0][:timesteps], label="action 0")
                plt.plot(np.arange(1, timesteps+1), num_actions[1][:timesteps], label="action 1")
                plt.plot(np.arange(1, timesteps+1), num_actions[2][:timesteps], label="no action")
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("time in recall phase")
            plt.ylabel("proportion of actions")
            plt.legend()
            plt.tight_layout()
            savefig(fig_path/"actions_distribution", fig_name)

        plot_actions(correct_trials, timestep_each_phase, "all")

        # actions for trials taking diffrent numbers of time steps
        for j in range(1, timestep_each_phase+1):
            chosen_trials = np.where(answer_timesteps == j)[0]
            plot_actions(chosen_trials, j, "timestep_{}".format(j))



        """ plot memories retrieved at each timestep """
        recall_probability = RecallProbabilityInTime()
        recall_probability.fit(np.repeat(np.arange(4).reshape(1,-1), retrieved_memories.shape[0], axis=0), 
                               retrieved_memories, mask=actions_mask)
        recall_probability.visualize(fig_path, timesteps=[0, 1, 2, 3])
        recall_probability.visualize_in_time(fig_path/"recall_prob_in_time", save_name="all")

        for i in range(1, timestep_each_phase+1):
            chosen_trials = np.where(answer_timesteps == i)[0]
            if len(chosen_trials) < 1:
                continue
            recall_probability.fit(np.repeat(np.arange(4).reshape(1,-1), chosen_trials.shape[0], axis=0),
                                      retrieved_memories[chosen_trials], mask=actions_mask[chosen_trials])
            recall_probability.visualize_in_time(fig_path/"recall_prob_in_time", time=i, save_name="timestep_{}".format(i))



        """ decoding answer """
        ridge_decoder = RidgeClassifier()
        decoder = DMAnswerDecoder(decoder=ridge_decoder)
        decoder.fit(c_recalling.transpose(1, 0, 2), answers)
        decoder.visualize(save_path=fig_path/"decode_answer", save_name="all_rec", xlabel="time in recall phase", figsize=(4, 3.3))

        decoder.fit(c_memorizing.transpose(1, 0, 2), answers)
        final_answer_decode_results = decoder.results
        decoder.visualize(save_path=fig_path/"decode_answer", save_name="all_enc", xlabel="time in encoding phase", figsize=(4, 3.3))

        for j in range(1, timestep_each_phase+1):
            chosen_trials = np.where(answer_timesteps == j)[0]
            if len(chosen_trials) < 10:
                continue
            decoder.fit(c_recalling[chosen_trials, :j].transpose(1, 0, 2), answers[chosen_trials])
            decoder.visualize(save_path=fig_path/"decode_answer", save_name="timestep_{}".format(j), xlabel="time in recall phase", figsize=(4, 3.3))


        """ decode each possible answer """
        # compute all possible answers
        num_features = env.unwrapped.num_features
        num_questions = num_features * (num_features - 1)
        all_possible_answers = np.zeros((correct_trials.shape[0], num_questions))
        cnt_trial = 0
        for k in correct_trials:
            mem_seq = data["trial_data"][k]["memory_sequence"]
            cnt = 0
            for i in range(num_features):
                for j in range(num_features):
                    if i == j:
                        continue
                    matched_trial = np.where(mem_seq[:, i] == 1)[0]
                    answer = np.sum(mem_seq[matched_trial, j]) % 2
                    all_possible_answers[cnt_trial, cnt] = answer
                    cnt += 1
            cnt_trial += 1

        plt.figure(figsize=(1.0*timestep_each_phase, 3.3), dpi=180)
        for i in range(num_questions):
            decoder.fit(c_memorizing.transpose(1, 0, 2), all_possible_answers[:, i])
            plt.plot(np.arange(1, timestep_each_phase+1), decoder.results, color="tab:blue")
        plt.plot(np.arange(1, final_answer_decode_results.shape[0]+1), final_answer_decode_results, color="tab:orange", label="final answer")
        plt.legend()
        plt.xlabel("time in encoding phase")
        plt.ylabel("answer decoding accuracy")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        savefig(fig_path/"decode_answer", "all_ans_all_enc")


        """ decoding """
        ridge_decoder = RidgeClassifier()
        ridge = ItemIdentityDecoder(decoder=ridge_decoder)
        os.makedirs(fig_path/"decode_data", exist_ok=True)

        # decode item identity from encoding phase, by time step
        ridge_encoding_item, _ = ridge.fit(c_memorizing.transpose(1, 0, 2), memory_sequences.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_item", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase", figsize=(4, 3.3))
        np.save(fig_path/"decode_data"/"ridge_encoding_item.npy", ridge_encoding_item)

        # decode related feature from encoding phase, by time step
        ridge_encoding_sum_feature, _ = ridge.fit(c_memorizing.transpose(1, 0, 2), sum_features.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_sumf", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase", figsize=(4, 3.3))
        np.save(fig_path/"decode_data"/"ridge_encoding_sum_feature.npy", ridge_encoding_sum_feature)

        # decode item identity and related feature from encoding phase, all time step together
        ridge_classifier = RidgeClassifier()
        classifier = Classifier(decoder=ridge_classifier)
        _, identity_acc = classifier.fit(c_memorizing.transpose(1, 0, 2), memory_sequences.transpose(1, 0))
        _, sum_feature_acc = classifier.fit(c_memorizing.transpose(1, 0, 2), sum_features.transpose(1, 0))
        print("identity_acc: ", identity_acc)
        print("sum_feature_acc: ", sum_feature_acc)
        with open(fig_path/"enc_ridge.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([identity_acc, sum_feature_acc])


        # decode item identity from recall phase, by time step, all trials
        ridge_recall_item, _ = ridge.fit(c_recalling.transpose(1, 0, 2), memory_sequences.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_item", colormap_label="item position\nin study order",
                                xlabel="time in recall phase", figsize=(4, 3.3))
        np.save(fig_path/"decode_data"/"ridge_decoding_item.npy", ridge_recall_item)

        # decode related feature from recall phase, by time step, all trials
        ridge_recall_sum_feature, _ = ridge.fit(c_recalling.transpose(1, 0, 2), sum_features.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_sumf", colormap_label="item position\nin study order",
                                xlabel="time in recall phase", figsize=(4, 3.3))
        np.save(fig_path/"decode_data"/"ridge_decoding_sum_feature.npy", ridge_recall_sum_feature)

        # decode item identity and related feature from recall phase, by time step, separate for trials with different answer timesteps
        for j in range(1, timestep_each_phase+1):
            chosen_trials = np.where(answer_timesteps == j)[0]
            if len(chosen_trials) < 10:
                continue

            ridge_recall_item, _ = ridge.fit(c_recalling[chosen_trials, :j].transpose(1, 0, 2), memory_sequences[chosen_trials].transpose(1, 0))
            ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_item_timestep_{}".format(j), colormap_label="item position\nin study order",
                                    xlabel="time in recall phase", figsize=(4, 3.3))
            np.save(fig_path/"decode_data"/"ridge_decoding_item_timestep_{}.npy".format(j), ridge_recall_item)

            ridge_recall_sum_feature, _ = ridge.fit(c_recalling[chosen_trials, :j].transpose(1, 0, 2), sum_features[chosen_trials].transpose(1, 0))
            ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_sumf_timestep_{}".format(j), colormap_label="item position\nin study order",
                                    xlabel="time in recall phase", figsize=(4, 3.3))
            np.save(fig_path/"decode_data"/"ridge_decoding_sum_feature_timestep_{}.npy".format(j), ridge_recall_sum_feature)

        
        # compute CRP curve
        recall_probability = RecallProbability()
        recall_probability.fit(memory_sequences, retrieved_memories)
        # plot CRP curve
        recall_probability.visualize_all_time(fig_path/"recall_prob")
        recall_probability.visualize(fig_path/"recall_prob")
        results_all_time = recall_probability.get_results_all_time()
        # write to csv file
        with open(fig_path/"recall_probability.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(results_all_time)

        # for j in range(1, timestep_each_phase+1):
        #     chosen_trials = np.where(answer_timesteps == j)[0]
        #     if len(chosen_trials) < 10:
        #         continue
        #     recall_probability.fit(memory_sequences[chosen_trials, :j], retrieved_memories[chosen_trials, :j])
        #     recall_probability.visualize_all_time(fig_path/"recall_prob_in_time", save_name="timestep_{}".format(j))
            

        # recall_probability_in_time = RecallProbabilityInTime()
        # recall_probability_in_time.fit(memory_sequences, retrieved_memories)
        # recall_probability_in_time.visualize_in_time(fig_path/"recal_prob_in_time", save_name="all")

        # for j in range(1, timestep_each_phase+1):
        #     chosen_trials = np.where(answer_timesteps == j)[0]
        #     if len(chosen_trials) < 1:
        #         continue
        #     recall_probability_in_time.fit(memory_sequences[chosen_trials, :j], retrieved_memories[chosen_trials, :j])
        #     recall_probability_in_time.visualize_in_time(fig_path/"recall_prob_in_time", save_name="timestep_{}".format(j))
