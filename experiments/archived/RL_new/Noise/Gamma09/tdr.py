import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge

from utils import savefig
from analysis.decomposition import TDR
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

        print(memory_contexts.shape, actions.shape, rewards.shape)

        if "ValueMemory" in readouts[0] and "similarity" in readouts[0]["ValueMemory"]:
            has_memory = True
        else:
            has_memory = False
        
        # print ground truths, actions and rewards for 5 trials
        for i in range(5):
            if has_memory:
                print("context {}, gt: {}, action: {}, retrieved memory: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=1)+1, rewards[i][env.memory_num:]))
            else:
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                rewards[i][env.memory_num:]))

        """ similarity of states """
        similarities = []
        for i in range(context_num):
            states = readouts[i]["state"].squeeze()
            similarity = skp.cosine_similarity(states, states)
            similarities.append(similarity)
        similarities = np.stack(similarities)
        similarity = np.mean(similarities, axis=0)

        plt.figure(figsize=(4, 3.3), dpi=180)
        plt.imshow(similarity[:timestep_each_phase, :timestep_each_phase], cmap="Blues")
        plt.colorbar(label="cosine similarity\nbetween hidden states")
        plt.xlabel("time in encoding phase")
        plt.ylabel("time in encoding phase")
        # plt.title("encoding-recalling state similarity")
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "encode")

        plt.figure(figsize=(4, 3.3), dpi=180)
        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, timestep_each_phase:timestep_each_phase*2], cmap="Blues")
        plt.colorbar(label="cosine similarity\nbetween hidden states")
        plt.xlabel("time in recall phase")
        plt.ylabel("time in recall phase")
        # plt.title("encoding-recalling state similarity")
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "recall")

        """ recall probability (output) (CRP curve) """
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        # plot CRP curve
        recall_probability.visualize_all_time(fig_path/"recall_prob")
        recall_probability.visualize(fig_path/"recall_prob")
        results_all_time = recall_probability.get_results_all_time()
        # write to csv file
        with open(fig_path/"recall_probability.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(results_all_time)

        """ TDR """
        tdr_context_num = 100
        # encoding phase
        states = []
        for i in range(tdr_context_num):
            states.append(readouts[i]['state'][:timestep_each_phase])
        states = np.stack(states).squeeze(2)
        tdr = TDR()
        # order_var_single_trial = np.arange(env.memory_num*2) % env.memory_num
        order_var_single_trial = np.arange(timestep_each_phase)
        order_var = np.repeat(order_var_single_trial.reshape(1, -1), tdr_context_num, axis=0)
        item_var = np.zeros((tdr_context_num, timestep_each_phase))
        for i in range(tdr_context_num):
            item_var[i, :timestep_each_phase] = memory_contexts[i]
            # item_var[i, timestep_each_phase:timestep_each_phase*2] = actions[i][timestep_each_phase:]
        task_var = np.stack((order_var, item_var))
        task_var = np.transpose(task_var, (1, 2, 0))
        var_labels = ["index", "item"]
        tdr_variance_enc = tdr.fit(states, task_var, var_labels)
        print("Encoding phase, TDR variance explained: ", tdr_variance_enc)

        # recall phase
        states = []
        for i in range(tdr_context_num):
            states.append(readouts[i]['state'][timestep_each_phase:timestep_each_phase*2])
        states = np.stack(states).squeeze(2)
        tdr = TDR()
        order_var_single_trial = np.arange(timestep_each_phase)
        order_var = np.repeat(order_var_single_trial.reshape(1, -1), tdr_context_num, axis=0)
        item_var = np.zeros((tdr_context_num, timestep_each_phase))
        for i in range(tdr_context_num):
            item_var[i, :timestep_each_phase] = actions[i][timestep_each_phase:]
        task_var = np.stack((order_var, item_var))
        task_var = np.transpose(task_var, (1, 2, 0))
        var_labels = ["index", "item"]
        tdr_variance_rec = tdr.fit(states, task_var, var_labels)
        print("Recall phase, TDR variance explained: ", tdr_variance_rec)

        tdr_variance = np.concatenate((tdr_variance_enc, tdr_variance_rec))
        np.save(fig_path/"tdr_variance.npy", tdr_variance)
