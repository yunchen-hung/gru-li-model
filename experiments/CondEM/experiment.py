import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import SVM
from analysis.behavior import RecallProbability, RecallProbabilityInTime
import sklearn.metrics.pairwise as skp


def run(data_all, model_all, env, paths, exp_name):
    run_num = len(list(data_all.keys()))

    for run_name, data in data_all.items():
        fig_path = paths["fig"]/run_name
        print()
        print(run_name)
        run_name_without_num = run_name.split("-")[0]

        model = model_all[run_name]

        """ compute the accuracy of the model, including accuracy, recall, and f1 score """
        tp_num = 0  # in correct answers, and answered
        fp_num = 0  # not in correct answers, but answered, including early stop
        # tn_num = 0  # not in correct answers, and not answered
        fn_num = 0  # in correct answers, but not answered
        for i in range(len(data["actions"])):
            actions = data["actions"][i]
            rewards = data["rewards"][i]
            trial_data = data["trial_data"][i]
            memory_num = len(trial_data["memory_sequence"])
            actions = actions[memory_num:]
            rewards = rewards[memory_num:]
            answered = np.zeros(len(trial_data["correct_answers"]), dtype=bool)
            for action, reward in zip(actions, rewards):
                if reward[0] == env.correct_reward:
                    tp_num += 1
                    for i, answer in enumerate(trial_data["correct_answers"]):
                        answer_int = env.convert_stimuli_to_action(answer)
                        if action == answer_int:
                            answered[i] = True
                            break
                elif action == env.action_space.n - 2:
                    pass
                else:
                    fp_num += 1
            fn_num += np.sum(~answered)
        print("tp_num", tp_num)
        print("fp_num", fp_num)
        print("fn_num", fn_num)
        accuracy = tp_num / (tp_num + fp_num)
        recall = tp_num / (tp_num + fn_num)
        f1 = 2 * accuracy * recall / (accuracy + recall)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1:", f1)


        """ PCA analysis """
        print(data['readouts'][0].keys())
        


        """ Decoding analysis """