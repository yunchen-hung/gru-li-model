import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_mutual_info_score

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import PCSelectivity, ItemIdentityDecoder, ItemIndexDecoder, Regressor, Classifier
from analysis.behavior import RecallProbability, RecallProbabilityInTime, TemporalFactor



def run(data_all, model_all, env, paths, exp_name):
    plt.rcParams['font.size'] = 16

    env = env[0]

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        # fig_path = paths["fig"]/run_name
        run_num = run_name.split("-")[-1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)
        print()
        print(run_name)

        data = data[0]
        model = model_all[run_name]

        sequence_len = env.unwrapped.sequence_len
        if hasattr(model, "step_for_each_timestep"):
            step_for_each_timestep = model.step_for_each_timestep
            timestep_each_phase = step_for_each_timestep * sequence_len
        else:
            step_for_each_timestep = 1
            timestep_each_phase = sequence_len

        # get recorded data and outputs of the model
        readouts = data['readouts']
        actions = data['actions']
        rewards = data['rewards']

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)

        # convert data to numpy array
        memory_contexts = []
        for i in range(all_context_num):
            memory_contexts.append(data['trial_data'][i]["memory_sequence_int"])
        memory_contexts = np.array(memory_contexts)     # ground truth of memory for each trial
        # memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        actions = np.array(actions).squeeze()                 # (trials, timesteps per trial)
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])        # (trials, timesteps per trial)

        print(memory_contexts.shape, actions.shape, rewards.shape)

        if "ValueMemory" in readouts[0] and "similarity" in readouts[0]["ValueMemory"]:
            has_memory = True
        else:
            has_memory = False
        
        # print ground truths, actions and rewards for 5 trials
        print("accuracy: {}".format(data['accuracy']))
        for i in range(5):
            if has_memory:
                print("context {}, gt: {}, action: {}, retrieved memory: {}, rewards: {}".format(i, memory_contexts[i], actions[i][sequence_len:], 
                np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=1)+1, rewards[i][sequence_len:]))
            else:
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][sequence_len:], 
                rewards[i][sequence_len:]))


        """ is there a bias to recall a particular item at a particular time step? """
        vocabulary_size = env.unwrapped.vocabulary_size
        recall_num_by_time = np.zeros((timestep_each_phase, vocabulary_size+1))
        rec_actions = actions[:, -timestep_each_phase:].astype(int)
        for i in range(context_num):
            for t in range(timestep_each_phase):
                recall_num_by_time[t][rec_actions[i][t]] += 1
        plt.figure(figsize=(0.25*vocabulary_size, 3.7), dpi=180)
        plt.imshow(recall_num_by_time, cmap="Blues")
        plt.colorbar(label="recall number")
        plt.xlabel("memory item")
        plt.ylabel("time in recall phase")
        plt.tight_layout()
        savefig(fig_path/"recall_num_by_time", "recall_num_by_time.png")



