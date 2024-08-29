import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ColorConverter
import seaborn as sns

from models.memory.similarity.lca import LCA
from utils import savefig


""" test color map """
# # cmap = plt.get_cmap('rainbow_r', 8)
# cmap = sns.color_palette("hls", 8)
# cmap = ListedColormap(cmap)
# norm = plt.Normalize(vmin=0.5, vmax=8.5)
# plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, 9))
# plt.show()
# plt.savefig("color_map.png")

plt.rcParams['font.size'] = 14


""" plot cogsci """

""" average plot of each temporal discount factor parameter """
exp_names = []
# exp_names.append("setup_gru_negmementreg")
# for i in range(1, 10):
#     exp_names.append("setup_gru_negmementreg_gamma0{}".format(i))
exp_names.append("setup_gru_negmementreg_gamma")

for exp in exp_names:
    run_names = []
    for i in range(100):
        # if i not in [3, 6, 7, 9, 15, 16, 19]:
            # print(i)
        run_names.append(exp + "-{}".format(i))

    ridge_encoding_res = []
    ridge_recall_res = []
    ridge_index_encoding_res = []
    ridge_index_recall_res = []
    pc_selectivity_res = []
    explained_var_res = []

    label = None
    for run_name in run_names:
        data = np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/pc_selectivity_encoding.npz".format(run_name), allow_pickle=True)
        if data['selectivity'][1][-1] > 0.8:
            ridge_encoding_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_encoding.npy".format(run_name)))
            ridge_recall_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_recall.npy".format(run_name)))
            ridge_index_encoding_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_encoding_index.npy".format(run_name)))
            ridge_index_recall_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_recall_index.npy".format(run_name)))
            pc_selectivity_res.append(data['selectivity'])
            explained_var_res.append(data['explained_var'])
        label = data['labels']

    ridge_encoding_res = np.mean(np.array(ridge_encoding_res), axis=0)
    ridge_recall_res = np.mean(np.array(ridge_recall_res), axis=0)
    ridge_index_encoding_std = np.std(ridge_index_encoding_res, axis=0)
    ridge_index_encoding_res = np.mean(np.array(ridge_index_encoding_res), axis=0)
    # print(ridge_index_encoding_std.shape, ridge_index_encoding_res.shape)
    ridge_index_recall_std = np.std(ridge_index_recall_res, axis=0)
    ridge_index_recall_res = np.mean(np.array(ridge_index_recall_res), axis=0)
    pc_selectivity_std = np.std(pc_selectivity_res, axis=0)
    pc_selectivity_res = np.mean(np.array(pc_selectivity_res), axis=0)
    # print(pc_selectivity_std.shape, pc_selectivity_res.shape)
    explained_var_std = np.std(explained_var_res, axis=0)
    explained_var_res = np.mean(np.array(explained_var_res), axis=0)

    # decoding plot of item identity in encoding phase
    plt.figure(figsize=(0.6 * ridge_encoding_res.shape[1], 3.3), dpi=180)
    n_steps = ridge_encoding_res.shape[0]
    # colors = sns.color_palette("viridis", n_steps+1)
    colors = ["#E76F51", "#EE8959", "#F4A261", "#E9C46A", "#8AB17D", "#2A9D8F", "#287271", "#264653"]
    for i in range(ridge_encoding_res.shape[1]):
        plt.plot(np.arange(1, ridge_encoding_res.shape[0]+1), ridge_encoding_res[:, i], label="item {}".format(i+1), color=colors[i])
    plt.xlim(0.5, 0.5 + ridge_encoding_res.shape[0])
    plt.xlabel("time in encoding phase")
    plt.ylabel("item identity\ndecoding accuracy")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cmap = ListedColormap(colors[:-1])
    norm = plt.Normalize(vmin=0.5, vmax=0.5+n_steps)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, n_steps+1), label="item index in study order")

    plt.tight_layout()
    savefig("./figures/{}".format(exp), "ridge_identity_encoding", format="svg")

    # decoding plot of item identity in recall phase
    plt.figure(figsize=(0.6 * ridge_recall_res.shape[1], 3.3), dpi=180)
    n_steps = ridge_recall_res.shape[0]
    # colors = sns.color_palette("viridis", n_steps+1)
    for i in range(ridge_recall_res.shape[1]):
        plt.plot(np.arange(1, ridge_recall_res.shape[0]+1), ridge_recall_res[:, i], label="item {}".format(i+1), color=colors[i])
    plt.xlim(0.5, 0.5 + ridge_recall_res.shape[0])
    plt.xlabel("time in recall phase")
    plt.ylabel("item identity\ndecoding accuracy")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cmap = ListedColormap(colors[:-1])
    norm = plt.Normalize(vmin=0.5, vmax=0.5+n_steps)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(1, n_steps+1), label="item index in recall order")

    plt.tight_layout()
    savefig("./figures/{}".format(exp), "ridge_identity_recall", format="svg")

    # decoding plot of item index in encoding and recall phase
    plt.figure(figsize=(0.5 * ridge_index_encoding_res.shape[0], 3.3), dpi=180)
    plt.plot(np.arange(1, ridge_index_encoding_res.shape[0]+1), ridge_index_encoding_res, c='tab:blue', label="encoding phase")
    plt.errorbar(np.arange(1, ridge_index_encoding_res.shape[0]+1), ridge_index_encoding_res, yerr=ridge_index_encoding_std, c='tab:blue', alpha=0.4)
    # plt.fill_between(np.arange(1, ridge_index_encoding_res.shape[0]+1), ridge_index_encoding_res - ridge_index_encoding_std,
    #                  ridge_index_encoding_res + ridge_index_encoding_std, color='tab:blue', alpha=0.2)
    plt.plot(np.arange(1, ridge_index_recall_res.shape[0]+1), ridge_index_recall_res, c='tab:orange', label="recall phase")
    plt.errorbar(np.arange(1, ridge_index_recall_res.shape[0]+1), ridge_index_recall_res, yerr=ridge_index_recall_std, c='tab:orange', alpha=0.4)
    # plt.fill_between(np.arange(1, ridge_index_recall_res.shape[0]+1), ridge_index_recall_res - ridge_index_recall_std,
    #                  ridge_index_recall_res + ridge_index_recall_std, color='tab:orange', alpha=0.2)
    plt.legend(fontsize=11)
    plt.xlim(0.5, ridge_index_encoding_res.shape[0]+0.5)
    plt.ylim(0.0, 1.05)
    plt.xlabel("time")
    plt.ylabel("item index\ndecoding accuracy")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    savefig("./figures/{}".format(exp), "ridge_index", format="svg")

    # decoding plot of item index in recall phase
    # plt.figure(figsize=(0.6 * ridge_index_recall_res.shape[0], 3.3), dpi=180)
    # plt.plot(np.arange(1, ridge_index_recall_res.shape[0]+1), ridge_index_recall_res)
    # plt.fill_between(np.arange(1, ridge_index_recall_res.shape[0]+1), ridge_index_recall_res - ridge_index_recall_std,
    #                  ridge_index_recall_res + ridge_index_recall_std, alpha=0.2)
    # plt.xlim(0.5, ridge_index_recall_res.shape[0]+0.5)
    # plt.ylim(0.0, 1.05)
    # plt.xlabel("hidden states in recall phase")
    # plt.ylabel("item index\ndecoding accuracy")

    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # plt.tight_layout()
    # savefig("./figures/{}".format(exp), "ridge_index_recall", format="svg")

    # pc selectivity
    plt.figure(figsize=(4, 3.3), dpi=180)
    c = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    label = label.item()
    label_names = ["item identity", "item index"]
    for i, label_name in enumerate(label_names):
        plt.plot(np.arange(1, 129), pc_selectivity_res[i], label=label_name, color=c[i])
        plt.fill_between(np.arange(1, 129), pc_selectivity_res[i] - pc_selectivity_std[i],
                        pc_selectivity_res[i] + pc_selectivity_std[i], color=c[i], alpha=0.2)
    plt.plot(np.arange(1, 129), explained_var_res, label="explained variance", color='k')
    plt.fill_between(np.arange(1, 129), explained_var_res - explained_var_std,
                    explained_var_res + explained_var_std, color='k', alpha=0.2)
    plt.xlabel("PC of hidden states")
    plt.ylabel("decoding accuracy")
    plt.legend(fontsize=11)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    savefig("./figures/{}".format(exp), "pc_selectivity", format="svg")




exp_names = []
temporal_discount_factors = []
exp_names.append("setup_gru_negmementreg")
temporal_discount_factors.append(0.0)
for i in range(1, 10):
    exp_names.append("setup_gru_negmementreg_gamma0{}".format(i))
    temporal_discount_factors.append(float(i)/10)
exp_names.append("setup_gru_negmementreg_gamma")
temporal_discount_factors.append(1.0)

accuracy_dict = {}
forward_asymmetry_dict = {}
temporal_factor_dict = {}
index_decoding_acc_dict = {}

for exp in exp_names:
    accuracy_dict[exp] = []
    forward_asymmetry_dict[exp] = []
    temporal_factor_dict[exp] = []
    index_decoding_acc_dict[exp] = []
    for i in range(20):
        run_name = exp + "-{}".format(i)
        with open("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/contiguity_effect.csv".format(run_name), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if float(row[0])>=0.65:
                    accuracy_dict[exp].append(float(row[0]))
                    forward_asymmetry_dict[exp].append(float(row[1]))
                    temporal_factor_dict[exp].append(float(row[2]))
                    data = np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/pc_selectivity_encoding.npz".format(run_name), allow_pickle=True)
                    index_decoding_acc_dict[exp].append(data['selectivity'][1][-1])


""" temporal discount factor - forward asymmetry & temporal factor """
mean_forward_asymmetry = []
std_forward_asymmetry = []
mean_temporal_factor = []
std_temporal_factor = []
mean_index_decoding_acc = []
std_index_decoding_acc = []
for exp in exp_names:
    mean_forward_asymmetry.append(np.mean(forward_asymmetry_dict[exp]))
    std_forward_asymmetry.append(np.std(forward_asymmetry_dict[exp]))
    mean_temporal_factor.append(np.mean(temporal_factor_dict[exp]))
    std_temporal_factor.append(np.std(temporal_factor_dict[exp]))
    mean_index_decoding_acc.append(np.mean(index_decoding_acc_dict[exp]))
    std_index_decoding_acc.append(np.std(index_decoding_acc_dict[exp]))
    


fig = plt.figure(figsize=(4.5, 3.3), dpi=180)
ax = fig.add_subplot(111)
ax.errorbar(temporal_discount_factors, mean_forward_asymmetry, yerr=std_forward_asymmetry, fmt='o',
            alpha=0.4, capsize=3, label="FA")
ax.set_xlabel("temporal discount factor ($\gamma$)")
ax.set_ylabel("forward asymmetry\n(FA)")

ax2 = ax.twinx()
ax2.errorbar(temporal_discount_factors, mean_temporal_factor, yerr=std_temporal_factor, fmt='o',  
             alpha=0.4, capsize=3, color='tab:orange', label="TF")
ax2.set_ylabel("temporal factor (TF)")

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.legend(loc=2, fontsize=11, bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes, framealpha=0.5)

plt.tight_layout()
savefig("./figures", "tdf_contiguity", format="svg")


fig = plt.figure(figsize=(4.3, 3.3), dpi=180)
plt.errorbar(temporal_discount_factors, mean_forward_asymmetry, yerr=std_forward_asymmetry, fmt='o',
            alpha=0.8, capsize=3)
plt.xlabel("temporal discount factor ($\gamma$)")
plt.ylabel("forward asymmetry (FA)")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "tdf_forw_asym", format="svg")


fig = plt.figure(figsize=(4.3, 3.3), dpi=180)
plt.errorbar(temporal_discount_factors, mean_temporal_factor, yerr=std_temporal_factor, fmt='o',  
             alpha=0.8, capsize=3, color='tab:orange')
plt.xlabel("temporal discount factor ($\gamma$)")
plt.ylabel("temporal factor (TF)")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "tdf_temporal_factor", format="svg")


fig = plt.figure(figsize=(4.2, 3.3), dpi=180)
plt.errorbar(temporal_discount_factors, mean_index_decoding_acc, yerr=std_index_decoding_acc, fmt='o',
            alpha=0.6, capsize=3, color='tab:green')
plt.xlabel("temporal discount factor ($\gamma$)")
plt.ylabel("decoding accuracy\nof item index")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "tdf_index_code", format="svg")



""" accuracy - forward asymmetry & temporal factor """
exp = "setup_gru_negmementreg_gamma06"

accuracy_list = []
forward_asymmetry_list = []
temporal_factor_list = []
index_decoding_acc_list = []

for i in range(100):
    run_name = exp + "-{}".format(i)
    with open("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/contiguity_effect.csv".format(run_name), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[0])>=0.65:
                accuracy_list.append(float(row[0]))
                forward_asymmetry_list.append(float(row[1]))
                temporal_factor_list.append(float(row[2]))
                data = np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/pc_selectivity_encoding.npz".format(run_name), allow_pickle=True)
                index_decoding_acc_list.append(data['selectivity'][1][-1])



# regressor = IsotonicRegression()
# # print(np.array(forward_asymmetry_list).shape, np.array(accuracy_list).shape)
# regressor.fit(np.array(forward_asymmetry_list), np.array(accuracy_list))
# # score = regressor.score(np.array(forward_asymmetry_list), np.array(accuracy_list))

# regressor = LinearRegression()
# regressor.fit(np.array(forward_asymmetry_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
# slope = regressor.coef_[0][0]
# intercept = regressor.intercept_[0]
# score = regressor.score(np.array(forward_asymmetry_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
# if slope*np.min(forward_asymmetry_list)+intercept > np.min(accuracy_list):
#     fit_line_point1 = [np.min(forward_asymmetry_list)*0.95, slope*np.min(forward_asymmetry_list)*0.95+intercept]
# else:
#     fit_line_point1 = [(np.min(accuracy_list)*0.95-intercept)/slope, np.min(accuracy_list)*0.95]
# if slope*np.max(forward_asymmetry_list)+intercept < np.max(accuracy_list):
#     fit_line_point2 = [np.max(forward_asymmetry_list)*1.05, slope*np.max(forward_asymmetry_list)*1.05+intercept]
# else:
#     fit_line_point2 = [(np.max(accuracy_list)*1.05-intercept)/slope, np.max(accuracy_list)*1.05]

plt.figure(figsize=(4, 3.3), dpi=180)
# forward_asymmetry_list = np.array(forward_asymmetry_list)
# forward_asymmetry_list -= 0.5
plt.scatter(forward_asymmetry_list, accuracy_list, alpha=0.5)

# plt.scatter([accuracy_list[9]], [forward_asymmetry_list[9]], color='k', alpha=0.5)   # model C
# plt.text(accuracy_list[9]-0.03, forward_asymmetry_list[9]-0.03, "1", fontsize=11)
# plt.scatter([accuracy_list[62]], [forward_asymmetry_list[62]], color='k', alpha=0.5)   # model B
# plt.text(accuracy_list[62]+0.02, forward_asymmetry_list[62]-0.02, "2", fontsize=11)
# plt.scatter([accuracy_list[14]], [forward_asymmetry_list[14]], color='k', alpha=0.5)   # model B
# plt.text(accuracy_list[14]+0.02, forward_asymmetry_list[14]-0.02, "3", fontsize=11)
# plt.scatter([accuracy_list[18]], [forward_asymmetry_list[18]], color='k', alpha=0.5)   # model A
# plt.text(accuracy_list[18]+0.02, forward_asymmetry_list[18]+0.02, "4", fontsize=11)

plt.scatter([forward_asymmetry_list[9]], [accuracy_list[9]], color='k', alpha=0.5)   # model C
plt.text(forward_asymmetry_list[9]-0.03, accuracy_list[9]-0.03, "1", fontsize=11)
plt.scatter([forward_asymmetry_list[62]], [accuracy_list[62]], color='k', alpha=0.5)   # model B
plt.text(forward_asymmetry_list[62]+0.02, accuracy_list[62]-0.02, "2", fontsize=11)
plt.scatter([forward_asymmetry_list[14]], [accuracy_list[14]], color='k', alpha=0.5)   # model B
plt.text(forward_asymmetry_list[14]+0.02, accuracy_list[14]-0.02, "3", fontsize=11)
plt.scatter([forward_asymmetry_list[18]], [accuracy_list[18]], color='k', alpha=0.5)   # model A
plt.text(forward_asymmetry_list[18]+0.02, accuracy_list[18]+0.02, "4", fontsize=11)

# plt.plot([fit_line_point1[0], fit_line_point2[0]], [fit_line_point1[1], fit_line_point2[1]], color='k', linestyle='--')
# plt.text((np.max(forward_asymmetry_list)+np.min(forward_asymmetry_list))/2, 0.98, "$r^2$={:.2f}".format(score), fontsize=12)
plt.xlabel("forward asymmetry")
plt.ylabel("task performance")
# plt.line = plt.plot([0.65, 1], [0, 0], color='k', linestyle='--')
# plt.text(0.87, -0.03, "random order", fontsize=11)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "acc_forw_asym", format="svg")


# regressor = IsotonicRegression()
# regressor.fit(np.array(temporal_factor_list), np.array(accuracy_list))
# # score = regressor.score(np.array(temporal_factor_list), np.array(accuracy_list))

# regressor = LinearRegression()
# regressor.fit(np.array(temporal_factor_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
# slope = regressor.coef_[0][0]
# intercept = regressor.intercept_[0]
# score = regressor.score(np.array(temporal_factor_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
# if slope*np.min(temporal_factor_list)+intercept > np.min(accuracy_list):
#     fit_line_point1 = [np.min(temporal_factor_list), slope*np.min(temporal_factor_list)+intercept]
# else:
#     fit_line_point1 = [(np.min(accuracy_list)-intercept)/slope, np.min(accuracy_list)]
# if slope*np.max(temporal_factor_list)+intercept < np.max(accuracy_list):
#     fit_line_point2 = [np.max(temporal_factor_list), slope*np.max(temporal_factor_list)+intercept]
# else:
#     fit_line_point2 = [(np.max(accuracy_list)-intercept)/slope, np.max(accuracy_list)]

plt.figure(figsize=(4, 3.3), dpi=180)
plt.scatter(temporal_factor_list, accuracy_list, alpha=0.5, color='tab:orange')

plt.scatter([temporal_factor_list[9]], [accuracy_list[9]], color='k', alpha=0.5)
plt.text(temporal_factor_list[9]-0.03, accuracy_list[9]-0.03, "1", fontsize=11)
plt.scatter([temporal_factor_list[62]], [accuracy_list[62]], color='k', alpha=0.5)
plt.text(temporal_factor_list[62]-0.02, accuracy_list[62]+0.02, "2", fontsize=11)
plt.scatter([temporal_factor_list[14]], [accuracy_list[14]], color='k', alpha=0.5)
plt.text(temporal_factor_list[14]-0.02, accuracy_list[14]+0.02, "3", fontsize=11)
plt.scatter([temporal_factor_list[18]], [accuracy_list[18]], color='k', alpha=0.5)
plt.text(temporal_factor_list[18]-0.02, accuracy_list[18]+0.02, "4", fontsize=11)

# plt.plot([fit_line_point1[0], fit_line_point2[0]], [fit_line_point1[1], fit_line_point2[1]], color='k', linestyle='--')
# plt.text((np.max(temporal_factor_list)+np.min(temporal_factor_list))/2, 0.98, "$r^2$={:.2f}".format(score), fontsize=11)
plt.ylabel("task performance")
plt.xlabel("temporal factor")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "acc_temporal_factor", format="svg")


plt.figure(figsize=(4, 3.5), dpi=180)
plt.scatter(index_decoding_acc_list, accuracy_list, alpha=0.5, color='tab:green')

plt.scatter([index_decoding_acc_list[9]], [accuracy_list[9]], color='k', alpha=0.5)
plt.text(index_decoding_acc_list[9]-0.03, accuracy_list[9]-0.03, "1", fontsize=11)
plt.scatter([index_decoding_acc_list[62]], [accuracy_list[62]], color='k', alpha=0.5)
plt.text(index_decoding_acc_list[62]-0.02, accuracy_list[62]+0.02, "2", fontsize=11)
plt.scatter([index_decoding_acc_list[14]], [accuracy_list[14]], color='k', alpha=0.5)
plt.text(index_decoding_acc_list[14]-0.02, accuracy_list[14]+0.02, "3", fontsize=11)
plt.scatter([index_decoding_acc_list[18]], [accuracy_list[18]], color='k', alpha=0.5)
plt.text(index_decoding_acc_list[18]-0.02, accuracy_list[18]+0.02, "4", fontsize=11)

plt.ylabel("task performance")
plt.xlabel("decoding accuracy\nof item index")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "acc_index_code", format="svg")


plt.figure(figsize=(4, 3.3), dpi=180)
colors = ["#F8961E", "#F9C74F", "#90BE6D", "#43AA8B"]

plt.scatter([temporal_factor_list[9]], [forward_asymmetry_list[9]], color=colors[0], alpha=0.75)   # model C
plt.text(temporal_factor_list[9]-0.03, forward_asymmetry_list[9]-0.03, "1", fontsize=11)
plt.scatter([temporal_factor_list[62]], [forward_asymmetry_list[62]], color=colors[1], alpha=0.75)   # model B
plt.text(temporal_factor_list[62]+0.02, forward_asymmetry_list[62]-0.02, "2", fontsize=11)
plt.scatter([temporal_factor_list[14]], [forward_asymmetry_list[14]], color=colors[2], alpha=0.75)   # model B
plt.text(temporal_factor_list[14]+0.02, forward_asymmetry_list[14]-0.02, "3", fontsize=11)
plt.scatter([temporal_factor_list[18]], [forward_asymmetry_list[18]], color=colors[3], alpha=0.75)   # model A
plt.text(temporal_factor_list[18]+0.02, forward_asymmetry_list[18]+0.02, "4", fontsize=11)

# plt.plot([fit_line_point1[0], fit_line_point2[0]], [fit_line_point1[1], fit_line_point2[1]], color='k', linestyle='--')
# plt.text((np.max(forward_asymmetry_list)+np.min(forward_asymmetry_list))/2, 0.98, "$r^2$={:.2f}".format(score), fontsize=12)
plt.ylabel("forward asymmetry (FA)")
plt.xlabel("temporal factor (TF)")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "sample4_tf_fa", format="svg")



""" recall probability multiple models """
exp_names = ["setup_gru_negmementreg_gamma06-9", "setup_gru_negmementreg_gamma06-62",
             "setup_gru_negmementreg_gamma06-14", "setup_gru_negmementreg_gamma06-18"]
recall_probs = []
for exp in exp_names:
    with open("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/recall_probability.csv".format(exp), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(len(row)):
                row[i] = float(row[i])
            recall_probs.append(np.array(row))
            break
nsteps = int((len(recall_probs[0])-1)/2) # 7

colors = ["#F8961E", "#F9C74F", "#90BE6D", "#43AA8B"]

plt.figure(figsize=(4.2, 3.3), dpi=180)
for i in range(len(exp_names)):
    plt.scatter(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c='w', marker="o", edgecolors=ColorConverter().to_rgba(colors[i], alpha=0.75), zorder=2)
    plt.plot(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c=colors[i], alpha=0.75, zorder=1, label="seed {}".format(i+1))
    plt.scatter(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c='w', marker="o", edgecolors=ColorConverter().to_rgba(colors[i], alpha=0.75), zorder=2)
    plt.plot(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c=colors[i], alpha=0.75, zorder=1)

    # plt.scatter(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c='w', marker="o", edgecolors=ColorConverter().to_rgba('k', alpha=1-i*0.25), zorder=2)
    # plt.plot(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c='k', alpha=1-i*0.25, label="seed {}".format(i+1), zorder=1)
    # plt.scatter(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c='w', marker="o", edgecolors=ColorConverter().to_rgba('k', alpha=1-i*0.25), zorder=2)
    # plt.plot(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c='k', alpha=1-i*0.25, zorder=1)
    # plt.scatter(np.array([0]), recall_probs[i][nsteps], c='r', marker="o", alpha=1-i*0.2)
plt.xlabel("lag")
plt.ylabel("conditional\nrecall probability")
    # title = title if title else "conditional recall probability"
    # plt.title(title)

plt.legend(fontsize=11)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "crp_different_colored", format="svg")


plt.figure(figsize=(4, 3.3), dpi=180)
plt.scatter(np.arange(-nsteps, 0), recall_probs[2][:nsteps], c='b', zorder=2)
plt.plot(np.arange(-nsteps, 0), recall_probs[2][:nsteps], c='k', zorder=1)
plt.scatter(np.arange(1, nsteps+1), recall_probs[2][nsteps+1:], c='b', zorder=2)
plt.plot(np.arange(1, nsteps+1), recall_probs[2][nsteps+1:], c='k', zorder=1)
plt.scatter(np.array([0]), recall_probs[2][nsteps], c='r')
plt.xlabel("lag")
plt.ylabel("conditional\nrecall probability")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "crp_seed2", format="svg")



recall_probs = []
exp = "setup_gru_negmementreg_gamma06"
i_valid = []
for i in range(100):
    run_name = exp + "-{}".format(i)
    with open("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/contiguity_effect.csv".format(run_name), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[0])>=0.65:
                i_valid.append(i)
                break
for i, i_v in enumerate(i_valid):
    run_name = exp + "-{}".format(i_v)
    if temporal_factor_list[i] < 0.45 or temporal_factor_list[i] > 0.6:
        continue
    with open("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/recall_probability.csv".format(run_name), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for k in range(len(row)):
                row[k] = float(row[k])
            recall_probs.append(np.array(row))
            break
recall_prob_avg = np.mean(np.array(recall_probs), axis=0)
recall_prob_std = np.std(np.array(recall_probs), axis=0)

plt.figure(figsize=(4, 3.3), dpi=180)

plt.scatter(np.arange(-nsteps, 0), recall_prob_avg[:nsteps], c='w', marker="o", edgecolors='k', zorder=2)
plt.plot(np.arange(-nsteps, 0), recall_prob_avg[:nsteps], c='k', zorder=1)
plt.errorbar(np.arange(-nsteps, 0), recall_prob_avg[:nsteps], yerr=recall_prob_std[:nsteps], c='k', alpha=0.5, zorder=1)
plt.scatter(np.arange(1, nsteps+1), recall_prob_avg[nsteps+1:], c='w', marker="o", edgecolors='k', zorder=2)
plt.plot(np.arange(1, nsteps+1), recall_prob_avg[nsteps+1:], c='k', zorder=1)
plt.errorbar(np.arange(1, nsteps+1), recall_prob_avg[nsteps+1:], yerr=recall_prob_std[nsteps+1:], c='k', alpha=0.5, zorder=1)

plt.xlabel("lag")
plt.ylabel("conditional recall probability")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "crp", format="svg")

