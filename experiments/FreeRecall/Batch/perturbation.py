import csv
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from utils import savefig
from analysis.decoding import ItemIdentityDecoder, ItemIndexDecoder
from analysis.perturbation import Perturbation


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

        print(memory_contexts.shape, actions.shape, rewards.shape)

        if "ValueMemory" in readouts[0] and "similarity" in readouts[0]["ValueMemory"]:
            has_memory = True
        else:
            has_memory = False


        """ get data for decoding """
        retrieved_memories = []
        for i in range(all_context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)
        retrieved_memories = np.stack(retrieved_memories)

        c_memorizing = np.stack([readouts[i]['state'][:timestep_each_phase].squeeze() for i in range(all_context_num)])   # context_num * time * state_dim
        c_recalling = np.stack([readouts[i]['state'][-timestep_each_phase:].squeeze() for i in range(all_context_num)])
        memory_sequence = np.stack([memory_contexts[i] for i in range(all_context_num)]) - 1    # context_num * time
        memory_sequence_onehot = np.eye(env.vocabulary_num)[memory_sequence]        # context_num * time * vocabulary_num
        actions_onehot = np.eye(env.vocabulary_num+1)[actions]                        # context_num * time * vocabulary_num

        memory_index = np.repeat(np.arange(env.memory_num).reshape(1, -1), all_context_num, axis=0)
        memory_index_onehot = np.eye(env.memory_num)[memory_index]

        action_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(env.memory_num):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    action_mask[i][t] = 1




        """ perturbation analysis """
        record_env = gym.vector.SyncVectorEnv([lambda: env])

        # baseline
        coef_identity = np.ones((env.vocabulary_num, model.hidden_dim))
        perturbation = Perturbation()
        identity_baseline = perturbation.fit(model, env, coef_identity, noise_start=0.0, noise_end=1.0, noise_step=0.05, trial_num=500)
        perturbation.visualize(fig_path / "perturbation", save_name="baseline_identity", 
                                xlabel="noise proportion", ylabel="accuracy", figsize=(4.0, 3.3))

        coef_index = np.ones((env.memory_num, model.hidden_dim))
        perturbation = Perturbation()
        index_baseline = perturbation.fit(model, env, coef_index, noise_start=0.0, noise_end=1.0, noise_step=0.05, trial_num=500)
        perturbation.visualize(fig_path / "perturbation", save_name="baseline_index",
                                xlabel="noise proportion", ylabel="accuracy", figsize=(4.0, 3.3))




        # fit the hidden state with item identity
        decoder = RidgeClassifier()
        xdata = c_memorizing.transpose(1, 0, 2)
        # ydata = memory_sequence.transpose(1, 0)
        ydata = memory_sequence_onehot.transpose(1, 0, 2)
        decoder.fit(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        score = decoder.score(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        print("identity decoding score:", score)
        coef = decoder.coef_
        # print(coef.shape)
        # print(np.sum(coef, axis=0))
        # print(coef[0])
    
        perturbation = Perturbation()
        identity_enc_all_res = perturbation.fit(model, env, coef, noise_start=0.0, noise_end=1.0, noise_step=0.05, trial_num=500)
        perturbation.visualize(fig_path / "perturbation", save_name="identity_enc_all", 
                                xlabel="noise proportion", ylabel="accuracy", figsize=(4.0, 3.3))
        
        decoder = RidgeClassifier()
        xdata = c_recalling[action_mask]
        print(xdata.shape)
        # ydata = memory_sequence.transpose(1, 0)
        ydata = actions_onehot[:, -timestep_each_phase:][action_mask]
        decoder.fit(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        score = decoder.score(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        print("identity decoding score:", score)
        coef = decoder.coef_
        # print(coef.shape)
        # print(np.sum(coef, axis=0))
        # print(coef[0])
    
        perturbation = Perturbation()
        identity_rec_all_res = perturbation.fit(model, env, coef, noise_start=0.0, noise_end=1.0, noise_step=0.05, trial_num=500)
        perturbation.visualize(fig_path / "perturbation", save_name="identity_rec_all", 
                                xlabel="noise proportion", ylabel="accuracy", figsize=(4.0, 3.3))



        # fit the hidden state with item index
        decoder = RidgeClassifier()
        xdata = c_memorizing.transpose(1, 0, 2)
        ydata = memory_index_onehot.transpose(1, 0, 2)
        decoder.fit(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        score = decoder.score(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        print("index decoding score:", score)
        coef = decoder.coef_

        perturbation = Perturbation()
        index_enc_all_res = perturbation.fit(model, env, coef, noise_start=0.0, noise_end=1.0, noise_step=0.05, trial_num=500)
        perturbation.visualize(fig_path / "perturbation", save_name="index_enc_all", 
                                xlabel="noise proportion", ylabel="accuracy", figsize=(4.0, 3.3))
        
        decoder = RidgeClassifier()
        xdata = c_recalling[action_mask]
        ydata = memory_index_onehot[action_mask]
        decoder.fit(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        score = decoder.score(xdata.reshape(-1, xdata.shape[-1]), ydata.reshape(-1, ydata.shape[-1]))
        print("index decoding score:", score)
        coef = decoder.coef_

        perturbation = Perturbation()
        index_rec_all_res = perturbation.fit(model, env, coef, noise_start=0.0, noise_end=1.0, noise_step=0.05, trial_num=500)
        perturbation.visualize(fig_path / "perturbation", save_name="index_rec_all", 
                                xlabel="noise proportion", ylabel="accuracy", figsize=(4.0, 3.3))



        figsize = (5.0, 3.3)
        plt.figure(figsize=figsize, dpi=180)

        x = np.arange(0.0, 1.0, 0.05)
        plt.plot(x, identity_baseline, label="baseline")
        plt.plot(x, identity_enc_all_res, label="decoder for\nencoding phase")
        plt.plot(x, identity_rec_all_res, label="decoder for\nrecall phase")

        plt.legend(fontsize=9)
        plt.xlabel("noise proportion")
        plt.ylabel("accuracy")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        savefig(fig_path / "perturbation", "identity", format="png")


        plt.figure(figsize=figsize, dpi=180)
        plt.figure(figsize=figsize, dpi=180)

        x = np.arange(0.0, 1.0, 0.05)
        plt.plot(x, index_baseline, label="baseline")
        plt.plot(x, index_enc_all_res, label="decoder for\nencoding phase")
        plt.plot(x, index_rec_all_res, label="decoder for\nrecall phase")

        plt.legend(fontsize=9)
        plt.xlabel("noise proportion")
        plt.ylabel("accuracy")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        savefig(fig_path / "perturbation", "index", format="png")

