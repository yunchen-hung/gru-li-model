import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ColorConverter
import seaborn as sns

from models.memory.similarity.lca import LCA
from utils import savefig


plt.rcParams['font.size'] = 14


results_dir = Path("./experiments/RL/Noise/figures/ValueMemoryGRU")


""" count accuracy of each hyperparam setting """
"""
for noise in ["0", "02", "04", "06", "08", "1"]:
    for gamma1 in ["00", "03", "06", "09"]:
        exp_name = "setup_noise{}_gamma{}".format(noise, gamma1)
        accuracy = 0.0
        forward_asymmetry = 0.0
        temporal_factor = 0.0
        cnt = 0
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.5:
                        cnt += 1
                        accuracy += float(row[0])
                        forward_asymmetry += float(row[1])
                        temporal_factor += float(row[2])
        if cnt == 0:
            cnt = 1
        print(exp_name, np.round(accuracy / cnt, 4), np.round(forward_asymmetry / cnt, 4), np.round(temporal_factor / cnt, 4))
    print()

    for gamma2 in ["00", "03", "06", "09"]:
        exp_name = "setup_noise{}_gamma{}_nostepbetween".format(noise, gamma2)
        accuracy = 0.0
        forward_asymmetry = 0.0
        temporal_factor = 0.0
        cnt = 0
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.5:
                        cnt += 1
                        accuracy += float(row[0])
                        forward_asymmetry += float(row[1])
                        temporal_factor += float(row[2])
        if cnt == 0:
            cnt = 1
        print(exp_name, np.round(accuracy / cnt, 4), np.round(forward_asymmetry / cnt, 4), np.round(temporal_factor / cnt, 4))
    print()
    print()
"""

def plot_bar(data, std, title, save_path):
    plt.figure(figsize=(6, 4), dpi=180)

    x = np.arange(0, 1.2, 0.2)
    plt.bar(x, data, yerr=std, width=0.1)
    plt.xticks(x, ["0", "0.2", "0.4", "0.6", "0.8", "1"])

    plt.xlabel("Noise proportion")
    plt.ylabel(title)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    savefig(save_path, title)

def plot_matrix(data, title, save_path):
    plt.figure(figsize=(6, 4), dpi=180)

    plt.imshow(data.T, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.xticks(np.arange(6), ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.yticks(np.arange(4), ["0", "0.3", "0.6", "0.9"])

    plt.xlabel("Noise proportion")
    plt.ylabel("Gamma")
    plt.title(title)

    plt.tight_layout()

    savefig(save_path, title)


""" contiguity effect and strategy changing with noise """
# for seqlen=8, barplot with error bars
# metrics: forward asymmetry, temporal factor, decoding accuracy of item index
# x-axis: noise proportion

threshold = 0.8

""" all models """
accuracy = []
forward_asym = []
temporal_fact = []
index_decoding_acc = []

for noise in ["0", "02", "04", "06", "08", "1"]:
    for gamma1 in ["00", "03", "06", "09"]:
        acc = []
        fa = []
        tf = []
        ida = []
        exp_name = "setup_noise{}_gamma{}".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    acc.append(float(row[0]))
                    fa.append(float(row[1]))
                    tf.append(float(row[2]))
                    data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                    if data['selectivity'][1][-1] >= threshold:
                        ida.append(1.0)
                    else:
                        ida.append(0.0)
        accuracy.append(np.mean(acc))
        forward_asym.append(np.mean(fa))
        temporal_fact.append(np.mean(tf))
        index_decoding_acc.append(np.mean(ida))

accuracy = np.array(accuracy).reshape(6, 4)
forward_asym = np.array(forward_asym).reshape(6, 4)
temporal_fact = np.array(temporal_fact).reshape(6, 4)
index_decoding_acc = np.array(index_decoding_acc).reshape(6, 4)

figpath = Path("./figures/seq8_all")
plot_matrix(accuracy, "performance", figpath)
plot_matrix(forward_asym, "forward asymmetry", figpath)
plot_matrix(temporal_fact, "temporal factor", figpath)
plot_matrix(index_decoding_acc, "index coding proportion", figpath)


accuracy = []
forward_asym = []
temporal_fact = []
index_decoding_acc = []

for noise in ["0", "02", "04", "06", "08", "1"]:
    for gamma1 in ["00", "03", "06", "09"]:
        acc = []
        fa = []
        tf = []
        ida = []
        exp_name = "setup_noise{}_gamma{}_nostepbetween".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    acc.append(float(row[0]))
                    fa.append(float(row[1]))
                    tf.append(float(row[2]))
                    data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                    if data['selectivity'][1][-1] >= threshold:
                        ida.append(1.0)
                    else:
                        ida.append(0.0)
        accuracy.append(np.mean(acc))
        forward_asym.append(np.mean(fa))
        temporal_fact.append(np.mean(tf))
        index_decoding_acc.append(np.mean(ida))

accuracy = np.array(accuracy).reshape(6, 4)
forward_asym = np.array(forward_asym).reshape(6, 4)
temporal_fact = np.array(temporal_fact).reshape(6, 4)
index_decoding_acc = np.array(index_decoding_acc).reshape(6, 4)

figpath = Path("./figures/seq8_all_nostep")
plot_matrix(accuracy, "performance", figpath)
plot_matrix(forward_asym, "forward asymmetry", figpath)
plot_matrix(temporal_fact, "temporal factor", figpath)
plot_matrix(index_decoding_acc, "index coding proportion", figpath)


""" all models with performance > 80 """
accuracy = []
forward_asym = []
temporal_fact = []
index_decoding_acc = []
accuracy_std = []
forward_asym_std = []
temporal_fact_std = []

for noise in ["0", "02", "04", "06", "08", "1"]:
    acc = []
    fa = []
    tf = []
    ida = []
    for gamma1 in ["00", "03", "06", "09"]:
        exp_name = "setup_noise{}_gamma{}".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.8 and float(row[2]) > 0.4:
                        acc.append(float(row[0]))
                        fa.append(float(row[1]))
                        tf.append(float(row[2]))
                        data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                        if data['selectivity'][1][-1] >= threshold:
                            ida.append(1.0)
                        else:
                            ida.append(0.0)
                        # ida.append(data['selectivity'][1][-1])
        exp_name = "setup_noise{}_gamma{}_nostepbetween".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.8 and float(row[2]) > 0.4:
                        acc.append(float(row[0]))
                        fa.append(float(row[1]))
                        tf.append(float(row[2]))
                        data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                        if data['selectivity'][1][-1] >= threshold:
                            ida.append(1.0)
                        else:
                            ida.append(0.0)
                        # ida.append(data['selectivity'][1][-1])
    accuracy.append(np.mean(acc))
    forward_asym.append(np.mean(fa))
    temporal_fact.append(np.mean(tf))
    index_decoding_acc.append(np.mean(ida))
    accuracy_std.append(np.std(acc))
    forward_asym_std.append(np.std(fa))
    temporal_fact_std.append(np.std(tf))
    
figpath = Path("./figures/seq8_perf80")
plot_bar(accuracy, accuracy_std, "performance", figpath)
plot_bar(forward_asym, forward_asym_std, "forward asymmetry", figpath)
plot_bar(temporal_fact, temporal_fact_std, "temporal factor", figpath)
plot_bar(index_decoding_acc, None, "index coding proportion", figpath)


""" models with best hyperparam settings """
accuracy = []
forward_asym = []
temporal_fact = []
index_decoding_acc = []
accuracy_std = []
forward_asym_std = []
temporal_fact_std = []
# index_decoding_acc_std = []

gamma = ["00", "00", "00", "03", "09", "09"]
follow_str = ["_nostepbetween", "_nostepbetween", "_nostepbetween", "_nostepbetween", "", ""]

for i, noise in enumerate(["0", "02", "04", "06", "08", "1"]):
    exp_name = "setup_noise{}_gamma{}{}".format(noise, gamma[i], follow_str[i])
    acc = []
    fa = []
    tf = []
    ida = []
    for j in range(100):
        with open(results_dir / exp_name / str(j) / "contiguity_effect.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                if float(row[0]) >= 0.8 and float(row[2]) > 0.4:
                    acc.append(float(row[0]))
                    fa.append(float(row[1]))
                    tf.append(float(row[2]))
                    data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, j), allow_pickle=True)
                    if data['selectivity'][1][-1] >= threshold:
                        ida.append(1.0)
                    else:
                        ida.append(0.0)
                    # ida.append(data['selectivity'][1][-1])
    accuracy.append(np.mean(acc))
    forward_asym.append(np.mean(fa))
    temporal_fact.append(np.mean(tf))
    index_decoding_acc.append(np.mean(ida))
    accuracy_std.append(np.std(acc))
    forward_asym_std.append(np.std(fa))
    temporal_fact_std.append(np.std(tf))

figpath = Path("./figures/seq8_bestparam")
plot_bar(accuracy, accuracy_std, "performance", figpath)
plot_bar(forward_asym, forward_asym_std, "forward asymmetry", figpath)
plot_bar(temporal_fact, temporal_fact_std, "temporal factor", figpath)
plot_bar(index_decoding_acc, None, "index coding proportion", figpath)



""" top 10% performing models """
accuracy = []
forward_asym = []
temporal_fact = []
index_decoding_acc = []
accuracy_std = []
forward_asym_std = []
temporal_fact_std = []


for noise in ["0", "02", "04", "06", "08", "1"]:
    acc = []
    fa = []
    tf = []
    ida = []
    for gamma1 in ["00", "03", "06", "09"]:
        exp_name = "setup_noise{}_gamma{}".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.8 and float(row[2]) > 0.4:
                        acc.append(float(row[0]))
                        fa.append(float(row[1]))
                        tf.append(float(row[2]))
                        data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                        if data['selectivity'][1][-1] >= threshold:
                            ida.append(1.0)
                        else:
                            ida.append(0.0)
                        # ida.append(data['selectivity'][1][-1])
        exp_name = "setup_noise{}_gamma{}_nostepbetween".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.8 and float(row[2]) > 0.4:
                        acc.append(float(row[0]))
                        fa.append(float(row[1]))
                        tf.append(float(row[2]))
                        data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                        if data['selectivity'][1][-1] >= threshold:
                            ida.append(1.0)
                        else:
                            ida.append(0.0)
    top = np.argsort(acc)[-min(20, len(acc)):]
    acc = np.array(acc)
    fa = np.array(fa)
    tf = np.array(tf)
    ida = np.array(ida)
    accuracy.append(np.mean(acc[top]))
    forward_asym.append(np.mean(fa[top]))
    temporal_fact.append(np.mean(tf[top]))
    index_decoding_acc.append(np.mean(ida[top]))
    accuracy_std.append(np.std(acc[top]))
    forward_asym_std.append(np.std(fa[top]))
    temporal_fact_std.append(np.std(tf[top]))

figpath = Path("./figures/seq8_top10")
plot_bar(accuracy, accuracy_std, "performance", figpath)
plot_bar(forward_asym, forward_asym_std, "forward asymmetry", figpath)
plot_bar(temporal_fact, temporal_fact_std, "temporal factor", figpath)
plot_bar(index_decoding_acc, None, "index coding proportion", figpath)


""" models with highest congituigy effect (?) """
accuracy = []
forward_asym = []
temporal_fact = []
index_decoding_acc = []
accuracy_std = []
forward_asym_std = []
temporal_fact_std = []


for noise in ["0", "02", "04", "06", "08", "1"]:
    acc = []
    fa = []
    tf = []
    ida = []
    for gamma1 in ["00", "03", "06", "09"]:
        exp_name = "setup_noise{}_gamma{}".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.8 and float(row[2]) > 0.4:
                        acc.append(float(row[0]))
                        fa.append(float(row[1]))
                        tf.append(float(row[2]))
                        data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                        if data['selectivity'][1][-1] >= threshold:
                            ida.append(1.0)
                        else:
                            ida.append(0.0)
                        # ida.append(data['selectivity'][1][-1])
        exp_name = "setup_noise{}_gamma{}_nostepbetween".format(noise, gamma1)
        for i in range(20):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.8 and float(row[2]) > 0.4:
                        acc.append(float(row[0]))
                        fa.append(float(row[1]))
                        tf.append(float(row[2]))
                        data = np.load("./experiments/RL/Noise/figures/ValueMemoryGRU/{}/{}/pc_selectivity_encoding.npz".format(exp_name, i), allow_pickle=True)
                        if data['selectivity'][1][-1] >= threshold:
                            ida.append(1.0)
                        else:
                            ida.append(0.0)
    top = np.argsort(tf)[-min(20, len(acc)):]
    acc = np.array(acc)
    fa = np.array(fa)
    tf = np.array(tf)
    ida = np.array(ida)
    accuracy.append(np.mean(acc[top]))
    forward_asym.append(np.mean(fa[top]))
    temporal_fact.append(np.mean(tf[top]))
    index_decoding_acc.append(np.mean(ida[top]))
    accuracy_std.append(np.std(acc[top]))
    forward_asym_std.append(np.std(fa[top]))
    temporal_fact_std.append(np.std(tf[top]))

figpath = Path("./figures/seq8_toptf10")
plot_bar(accuracy, accuracy_std, "performance", figpath)
plot_bar(forward_asym, forward_asym_std, "forward asymmetry", figpath)
plot_bar(temporal_fact, temporal_fact_std, "temporal factor", figpath)
plot_bar(index_decoding_acc, None, "index coding proportion", figpath)


