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
        timestep_each_phase = env.sequence_len

        readouts = data['readouts']


        if "mem_gate_recall" in readouts[0]:
            plt.figure(figsize=(4, 3), dpi=180)
            for i in range(context_num):
                em_gates = readouts[i]['mem_gate_recall']
                plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[:timestep_each_phase], label="context {}".format(i))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("time of recall phase")
            plt.ylabel("memory gate")
            plt.tight_layout()
            savefig(fig_path, "em_gate_recall")

        

        