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

        readouts = data['readouts']

        """ compute the accuracy of the model, including accuracy, recall, and f1 score """
        # tp_num = 0  # in correct answers, and answered
        # fp_num = 0  # not in correct answers, but answered, including early stop
        # # tn_num = 0  # not in correct answers, and not answered
        # fn_num = 0  # in correct answers, but not answered
        # for i in range(context_num):
        #     actions = data["actions"][i]
        #     rewards = data["rewards"][i]
        #     trial_data = data["trial_data"][i]
        #     memory_num = len(trial_data["memory_sequence"])
        #     actions = actions[memory_num:]
        #     rewards = rewards[memory_num:]
        #     answered = np.zeros(len(trial_data["correct_answers"]), dtype=bool)
        #     for action, reward in zip(actions, rewards):
        #         if reward[0] == env.correct_reward:
        #             tp_num += 1
        #             for i, answer in enumerate(trial_data["correct_answers"]):
        #                 answer_int = env.convert_stimuli_to_action(answer)
        #                 if action == answer_int:
        #                     answered[i] = True
        #                     break
        #         elif action == env.action_space.n - 2:
        #             pass
        #         else:
        #             fp_num += 1
        #     fn_num += np.sum(~answered)
        # print("tp_num", tp_num)
        # print("fp_num", fp_num)
        # print("fn_num", fn_num)
        # accuracy = tp_num / (tp_num + fp_num)
        # recall = tp_num / (tp_num + fn_num)
        # f1 = 2 * accuracy * recall / (accuracy + recall)
        # print("Accuracy:", accuracy)
        # print("Recall:", recall)
        # print("F1:", f1)


        """ compute retrieved memory index """
        retrieved_memories = []
        for i in range(context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)


        """ amout of different actions """
        # correct action : no action: wrong action
        # correct action : not recalled correct action
        # before ending, correct action : recalled a not-correct item in a list but didn't output : other wrong actions
        correct_actions_num = 0
        not_know_actions_num = 0
        wrong_actions_num = 0
        not_recalled_correct_actions_num = 0
        recalled_but_not_output_correct_actions_num = 0

        for i in range(len(data["actions"])):
            actions = data["actions"][i]
            rewards = data["rewards"][i]
            trial_data = data["trial_data"][i]
            memory_num = len(trial_data["memory_sequence"])
            actions = actions[memory_num:]
            rewards = rewards[memory_num:]

            # remove not-know and stop actions at the end of the trial
            action_valid_index = len(actions)
            if actions[action_valid_index-1] == env.action_space.n - 1:
                action_valid_index -= 1
            if action_valid_index == 0:
                continue
            actions = actions[:action_valid_index]
            rewards = rewards[:action_valid_index]
            
            recalled = np.zeros(len(trial_data["memory_sequence"]), dtype=bool)
            answered = np.zeros(len(trial_data["correct_answers"]), dtype=bool)
            for t, (action, reward) in enumerate(zip(actions, rewards)):
                if reward[0] == env.correct_reward:
                    correct_actions_num += 1
                    for i, answer in enumerate(trial_data["correct_answers"]):
                        answer_int = env.convert_stimuli_to_action(answer)
                        if action == answer_int:
                            answered[i] = True
                            break
                elif action == env.action_space.n - 2:
                    try:
                        if not recalled[retrieved_memories[i][t]]:
                            recalled_but_not_output_correct_actions_num += 1
                    except:
                        pass
                    not_know_actions_num += 1
                else:
                    wrong_actions_num += 1
                try:
                    recalled[retrieved_memories[i][t]] = True
                except:
                    pass
            not_recalled_correct_actions_num += np.sum(~answered)

        with open(fig_path/"actions.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([correct_actions_num, not_know_actions_num, wrong_actions_num, not_recalled_correct_actions_num, recalled_but_not_output_correct_actions_num])

        correct_not_know_wrong = np.array([correct_actions_num, not_know_actions_num, wrong_actions_num])
        correct_not_know_wrong = correct_not_know_wrong / np.sum(correct_not_know_wrong)
        correct_not_recalled = np.array([correct_actions_num, not_recalled_correct_actions_num])
        correct_not_recalled = correct_not_recalled / np.sum(correct_not_recalled)
        correct_not_output_not_know_wrong = np.array([correct_actions_num, recalled_but_not_output_correct_actions_num, 
                                                      not_know_actions_num-recalled_but_not_output_correct_actions_num, wrong_actions_num])
        correct_not_output_not_know_wrong = correct_not_output_not_know_wrong / np.sum(correct_not_output_not_know_wrong)

        group1.append(correct_not_know_wrong)
        group2.append(correct_not_recalled)
        group3.append(correct_not_output_not_know_wrong)

    group1 = np.stack(group1)
    group1 = np.mean(group1, axis=0)
    group2 = np.stack(group2)
    group2 = np.mean(group2, axis=0)
    group3 = np.stack(group3)
    group3 = np.mean(group3, axis=0)

    plt.figure(figsize=(2, 4), dpi=180)
    bottom = np.zeros(group1.shape)
    labels = ["correct", "not know", "wrong"]
    for i, data in enumerate(group1):
        plt.bar(np.arange(1), [data], bottom=bottom, label=labels[i])
        bottom += data
    plt.legend()
    savefig(fig_path, "actions1")

    bottom = np.zeros(group2.shape)
    labels = ["correct", "not recalled"]
    for i, data in enumerate(group2):
        plt.bar(np.arange(1), [data], bottom=bottom, label=labels[i])
        bottom += data
    plt.legend()
    savefig(fig_path, "actions2")

    bottom = np.zeros(group3.shape)
    labels = ["correct", "recalled but not output", "not know", "wrong"]
    for i, data in enumerate(group3):
        plt.bar(np.arange(1), [data], bottom=bottom, label=labels[i])
        bottom += data
    plt.legend()
    savefig(fig_path, "actions3")

