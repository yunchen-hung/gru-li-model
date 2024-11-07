import json
from pathlib import Path
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

from models.memory.similarity.lca import LCA
from models.utils import softmax
from tasks import ConditionalEMRecall , MetaLearningEnv, ConditionalQuestionAnswer, FreeRecallRepeat, \
    FreeRecall, PlaceHolderWrapper

from utils import load_dict, savefig



""" plots for conditional question answer """
plt.rcParams['font.size'] = 14

# training curve
# training_acc_encq = np.load("./experiments/CondQA/Batch/saved_models/DeepValueMemoryGRU/setup_encq_earlystop-1/accuracy_1.npy")
# print(training_acc_encq.shape)

# training_acc_recq = np.load("./experiments/CondQA/Batch/saved_models/DeepValueMemoryGRU/setup_recq_earlystop-0/accuracy_1.npy")
# print(training_acc_recq.shape)

colors = [sns.color_palette()[0], sns.color_palette()[1]]

# accuracy
accs_encq = []
accs_recq = []
for i in range(3):
    with open("./experiments/CondQA/ResourceConstraint/figures/DeepValueMemoryGRU/setup_nonoise_encq/{}/accuracy.csv".format(i), "r") as f:
        accs_encq.append(np.array(f.read().split(",")[0], dtype=float))
    with open("./experiments/CondQA/ResourceConstraint/figures/DeepValueMemoryGRU/setup_nonoise/{}/accuracy.csv".format(i), "r") as f:
        accs_recq.append(np.array(f.read().split(",")[0], dtype=float))
accs_encq = np.array(accs_encq)
accs_recq = np.array(accs_recq)

mean_acc_encq = np.mean(accs_encq)
std_acc_encq = np.std(accs_encq)
mean_acc_recq = np.mean(accs_recq)
std_acc_recq = np.std(accs_recq)

def bar_plot(mean, std, baseline, ylabel, filename, figsize=(3, 3.3), baseline_text_pos="bottom"):
    plt.figure(figsize=figsize, dpi=200)
    plt.bar(["Before", "After"], mean, yerr=std, color=colors, capsize=5)
    plt.axhline(baseline, color='black', linestyle='--')
    if baseline_text_pos == "top":
        plt.text(1.5, baseline+0.005, "chance level", fontsize=12, ha='right', va='bottom')
    else:
        plt.text(1.5, baseline-0.015, "chance level", fontsize=12, ha='right', va='top')
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig("./figures_qa", filename, format="svg")

bar_plot([mean_acc_encq, mean_acc_recq], [std_acc_encq, std_acc_recq], 0.5, "Task performance", "performance")

# decoding results
identity_accs_encq = []
identity_accs_recq = []
feaure_accs_encq = []
feaure_accs_recq = []

for i in range(3):
    with open("./experiments/CondQA/ResourceConstraint/figures/DeepValueMemoryGRU/setup_nonoise_encq/{}/enc_ridge.csv".format(i), "r") as f:
        data = f.read().split(",")
        identity_accs_encq.append(data[0])
        feaure_accs_encq.append(data[1])
    with open("./experiments/CondQA/ResourceConstraint/figures/DeepValueMemoryGRU/setup_nonoise/{}/enc_ridge.csv".format(i), "r") as f:
        data = f.read().split(",")
        identity_accs_recq.append(data[0])
        feaure_accs_recq.append(data[1])

identity_accs_encq_mean = np.mean(np.array(identity_accs_encq, dtype=float))
identity_accs_encq_std = np.std(np.array(identity_accs_encq, dtype=float))
identity_accs_recq_mean = np.mean(np.array(identity_accs_recq, dtype=float))
identity_accs_recq_std = np.std(np.array(identity_accs_recq, dtype=float))
feature_accs_encq_mean = np.mean(np.array(feaure_accs_encq, dtype=float))
feature_accs_encq_std = np.std(np.array(feaure_accs_encq, dtype=float))
feature_accs_recq_mean = np.mean(np.array(feaure_accs_recq, dtype=float))
feature_accs_recq_std = np.std(np.array(feaure_accs_recq, dtype=float))

bar_plot([identity_accs_encq_mean, identity_accs_recq_mean], [identity_accs_encq_std, identity_accs_recq_std], 1.0/16, 
"Item decoding accuracy", "identity_decoding", baseline_text_pos="top")

bar_plot([feature_accs_encq_mean, feature_accs_recq_mean], [feature_accs_encq_std, feature_accs_recq_std], 0.5,
"Feature decoding accuracy", "feature_decoding")