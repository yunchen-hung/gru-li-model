import os
import csv
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_mutual_info_score

from train.criterions.rl import pick_action

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import PCSelectivity, ItemIdentityDecoder, ItemIndexDecoder, Regressor, Classifier, MultiRegressor, CrossClassifier
from analysis.behavior import RecallProbability, RecallProbabilityInTime, TemporalFactor



def compute_perturbed_performance(model, env, num_trials=1000):
    data = []
    for _ in range(num_trials):
        env.reset()
        data.append(env.unwrapped.memory_sequence_index)
    sequence_len = env.unwrapped.sequence_len

    """ original performance """
    states_all, actions_all = [], []
    for sequence in data:
        # Reset environment and model state
        obs_, info =env.reset(memory_sequence_index=sequence)
        obs = torch.Tensor(obs_).reshape(1, -1)
        done = False
        model.reset_memory()
        state = model.init_state(1)

        loss_masks, rewards = [], []
        states, actions = [], []
        memory_num = 0
        correct_actions, wrong_actions, not_know_actions = 0, 0, 0
        while not done:
            # set up the phase of the model 
            if info["phase"] == "encoding":
                model.set_encoding(True)
                model.set_retrieval(False)
                memory_num += 1
            elif info["phase"] == "recall":
                model.set_encoding(False)
                model.set_retrieval(True)
            # reset state between phases
            if "reset_state" in info and info["reset_state"]:
                state = model.init_state(1, recall=True, prev_state=state)
            
            output, value, state, _ = model(obs, state)
            action_distribution = output
            action, log_prob_action, action_max = pick_action(action_distribution)
            obs_, reward, _, _, info = env.step(action_max.cpu().detach().numpy().squeeze(axis=1))
            obs = torch.Tensor(obs_).reshape(1, -1)

            loss_masks.append([info["loss_mask"] and not done])
            rewards.append([reward])
            states.append(state)
            actions.append(action_max)
            done = info["done"]
            correct_actions += np.sum(info["correct"])
            wrong_actions += np.sum(info["wrong"])
            not_know_actions += np.sum(info["not_know"])

        states_all.append(states)
        actions_all.append(actions)

        original_performance = correct_actions / (correct_actions + wrong_actions + not_know_actions)

    """ group states and actions by index and identity respectively """
    states_all = np.array(states_all)
    actions_all = np.array(actions_all)

    

    """ perturbed performance """
    for i, sequence in enumerate(data):
        pass



def run(data_all, model_all, env, paths, exp_name, checkpoints=None, **kwargs):
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




        