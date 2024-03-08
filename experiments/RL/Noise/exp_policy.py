import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import PCSelectivity, ItemIdentityDecoder, ItemIndexDecoder
from analysis.behavior import RecallProbability, RecallProbabilityInTime, TemporalFactor



def run(data_all, model_all, env, paths, exp_name):
    plt.rcParams['font.size'] = 14

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        # fig_path = paths["fig"]/run_name
        run_num = run_name.split("-")[-1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)
        print()
        print(run_name)

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
        memory_contexts = np.array(data['trial_data'])     # ground truth of memory for each trial
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        actions = np.array(actions).squeeze(-1)
        # print(actions.shape)
        actions = actions.reshape(-1, actions.shape[-1])        # (trials, timesteps per trial)
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])        # (trials, timesteps per trial)


        """ policy distribution over all memory items """
        policy = []
        for i in range(all_context_num):
            policy.append(readouts[i]['decision'])
        policy = np.stack(policy).squeeze()
        print(policy.shape, actions.shape)

        policy_sorted = []
        for i in range(all_context_num):
            policy_sorted.append(policy[i, -env.memory_num:, actions[i, -env.memory_num:]])
        policy_sorted = np.stack(policy_sorted).squeeze()
        print(policy_sorted.shape)

        plt.imshow(policy_sorted[0], cmap="Blues")
        plt.colorbar()
        plt.title("policy distribution, one trial")
        plt.xlabel("time step")
        plt.ylabel("memory item")
        plt.tight_layout()
        savefig(fig_path/"policy", "one_trial.png")

        plt.imshow(np.mean(policy_sorted, axis=0), cmap="Blues")
        plt.colorbar()
        plt.title("policy distribution, averaged")
        plt.xlabel("time step")
        plt.ylabel("memory item")
        plt.tight_layout()
        savefig(fig_path/"policy", "all_trial.png")
