import numpy as np
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import SVM
from analysis.visualization import RecallProbability
import sklearn.metrics.pairwise as skp


def run(data_all, model_all, env, paths, exp_name):
    run_num = len(list(data_all.keys()))
    sim_mem, sim_enc, sim_rec, sim_enc_rec = [], [], [], []

    sim_mem = {}
    sim_enc = {}
    sim_rec = {}
    sim_enc_rec = {}
    accuracy = {}
    rec_prob = {}
    rec_prob_mem = {}
    rec_prob_all = {}
    rec_prob_mem_all = {}

    run_names_without_num = []
    for run_name in data_all.keys():
        run_name_without_num = run_name.split("-")[0]
        run_names_without_num.append(run_name_without_num)
        if run_name_without_num not in sim_mem.keys():
            sim_mem[run_name_without_num] = []
            sim_enc[run_name_without_num] = []
            sim_rec[run_name_without_num] = []
            sim_enc_rec[run_name_without_num] = []
            accuracy[run_name_without_num] = []
            rec_prob[run_name_without_num] = []
            rec_prob_mem[run_name_without_num] = []
            rec_prob_all[run_name_without_num] = []
            rec_prob_mem_all[run_name_without_num] = []

    for run_name, data in data_all.items():
        fig_path = paths["fig"]/run_name
        print()
        print(run_name)
        run_name_without_num = run_name.split("-")[0]

        model = model_all[run_name]
        if hasattr(model, "step_for_each_timestep"):
            step_for_each_timestep = model.step_for_each_timestep
            timestep_each_phase = step_for_each_timestep * env.memory_num
        else:
            step_for_each_timestep = 1
            timestep_each_phase = env.memory_num
        # timestep_each_phase = env.memory_num

        readouts = data['readouts']
        actions = data['actions']
        probs = data['probs']
        rewards = data['rewards']
        values = data['values']
        accuracy[run_name_without_num].append(data['accuracy'])

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)
        trial_num = len(actions[0])

        memory_contexts = np.array(data['memory_contexts'])
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])
        actions = np.array(actions)
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])
        
        for i in range(5):
            print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                rewards[i][env.memory_num:]))

