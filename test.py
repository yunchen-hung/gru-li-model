import csv
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ColorConverter
import seaborn as sns

from models.memory.similarity.lca import LCA
from models.utils import softmax
from tasks import ConditionalEMRecall, MetaLearningEnv
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
exp_names = []
temporal_discount_factors = []
exp_names.append("setup_gru_negmementreg")
temporal_discount_factors.append(0.0)
for i in range(1, 10):
    exp_names.append("setup_gru_negmementreg_gamma0{}".format(i))
    temporal_discount_factors.append(float(i)/10)
exp_names.append("setup_gru_negmementreg_gamma")
temporal_discount_factors.append(1.0)

for exp in exp_names:
    run_names = []
    for i in range(20):
        run_names.append(exp + "-{}".format(i))

    ridge_encoding_res = []
    ridge_recall_res = []
    ridge_index_encoding_res = []
    ridge_index_recall_res = []
    pc_selectivity_res = []
    explained_var_res = []

    label = None
    for run_name in run_names:
        ridge_encoding_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_encoding.npy".format(run_name)))
        ridge_recall_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_recall.npy".format(run_name)))
        ridge_index_encoding_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_encoding_index.npy".format(run_name)))
        ridge_index_recall_res.append(np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/ridge_recall_index.npy".format(run_name)))
        data = np.load("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/pc_selectivity_encoding.npz".format(run_name), allow_pickle=True)
        pc_selectivity_res.append(data['selectivity'])
        explained_var_res.append(data['explained_var'])
        label = data['labels']

    ridge_encoding_res = np.mean(np.array(ridge_encoding_res), axis=0)
    ridge_recall_res = np.mean(np.array(ridge_recall_res), axis=0)
    ridge_index_encoding_res = np.mean(np.array(ridge_index_encoding_res), axis=0)
    ridge_index_encoding_std = np.std(ridge_index_encoding_res, axis=0)
    ridge_index_recall_res = np.mean(np.array(ridge_index_recall_res), axis=0)
    ridge_index_recall_std = np.std(ridge_index_recall_res, axis=0)
    pc_selectivity_res = np.mean(np.array(pc_selectivity_res), axis=0)
    pc_selectivity_std = np.std(pc_selectivity_res, axis=0)
    explained_var_res = np.mean(np.array(explained_var_res), axis=0)
    explained_var_std = np.std(explained_var_res, axis=0)

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
        # plt.fill_between(np.arange(1, 129), pc_selectivity_res[i] - pc_selectivity_std[i],
        #                 pc_selectivity_res[i] + pc_selectivity_std[i], color=c[i], alpha=0.2)
    plt.plot(np.arange(1, 129), explained_var_res, label="explained variance", color='k')
    # plt.fill_between(np.arange(1, 129), explained_var_res - explained_var_std,
    #                 explained_var_res + explained_var_std, color='k', alpha=0.2)
    plt.xlabel("PC of hidden states")
    plt.ylabel("decoding accuracy")
    plt.legend(fontsize=11)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    savefig("./figures/{}".format(exp), "pc_selectivity", format="svg")




accuracy_dict = {}
forward_asymmetry_dict = {}
temporal_factor_dict = {}
accuracy_list = []
forward_asymmetry_list = []
temporal_factor_list = []

for exp in exp_names:
    accuracy_dict[exp] = []
    forward_asymmetry_dict[exp] = []
    temporal_factor_dict[exp] = []
    for i in range(20):
        run_name = exp + "-{}".format(i)
        with open("./experiments/RL/figures/cogsci/ValueMemoryGRU/{}/contiguity_effect.csv".format(run_name, run_name), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if float(row[0])>=0.65:
                    accuracy_dict[exp].append(float(row[0]))
                    forward_asymmetry_dict[exp].append(float(row[1]))
                    temporal_factor_dict[exp].append(float(row[2]))
    accuracy_list.extend(accuracy_dict[exp])
    forward_asymmetry_list.extend(forward_asymmetry_dict[exp])
    temporal_factor_list.extend(temporal_factor_dict[exp])


# temporal discount factor - forward asymmetry & temporal factor
mean_forward_asymmetry = []
std_forward_asymmetry = []
mean_temporal_factor = []
std_temporal_factor = []
for exp in exp_names:
    mean_forward_asymmetry.append(np.mean(forward_asymmetry_dict[exp]))
    std_forward_asymmetry.append(np.std(forward_asymmetry_dict[exp]))
    mean_temporal_factor.append(np.mean(temporal_factor_dict[exp]))
    std_temporal_factor.append(np.std(temporal_factor_dict[exp]))


plt.figure(figsize=(4, 3.3), dpi=180)
plt.errorbar(temporal_discount_factors, mean_forward_asymmetry, yerr=std_forward_asymmetry, fmt='o', capsize=3)
plt.xlabel("temporal discount factor")
plt.ylabel("forward asymmetry")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "tdf_forw_asym", format="svg")


plt.figure(figsize=(4, 3.3), dpi=180)
plt.errorbar(temporal_discount_factors, mean_temporal_factor, yerr=std_temporal_factor, fmt='o', capsize=3, color='tab:green')
plt.xlabel("temporal discount factor")
plt.ylabel("temporal factor")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "tdf_temporal_factor", format="svg")


# accuracy - forward asymmetry & temporal factor
regressor = LinearRegression()
regressor.fit(np.array(forward_asymmetry_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
slope = regressor.coef_[0][0]
intercept = regressor.intercept_[0]
score = regressor.score(np.array(forward_asymmetry_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
if slope*np.min(forward_asymmetry_list)+intercept > np.min(accuracy_list):
    fit_line_point1 = [np.min(forward_asymmetry_list)*0.95, slope*np.min(forward_asymmetry_list)*0.95+intercept]
else:
    fit_line_point1 = [(np.min(accuracy_list)*0.95-intercept)/slope, np.min(accuracy_list)*0.95]
if slope*np.max(forward_asymmetry_list)+intercept < np.max(accuracy_list):
    fit_line_point2 = [np.max(forward_asymmetry_list)*1.05, slope*np.max(forward_asymmetry_list)*1.05+intercept]
else:
    fit_line_point2 = [(np.max(accuracy_list)*1.05-intercept)/slope, np.max(accuracy_list)*1.05]

plt.figure(figsize=(4, 3.3), dpi=180)
plt.scatter(forward_asymmetry_list, accuracy_list, alpha=0.5)
plt.plot([fit_line_point1[0], fit_line_point2[0]], [fit_line_point1[1], fit_line_point2[1]], color='k', linestyle='--')
plt.text((np.max(forward_asymmetry_list)+np.min(forward_asymmetry_list))/2, 0.98, "R2={:.2f}".format(score), fontsize=11)
plt.ylabel("task accuracy")
plt.xlabel("forward asymmetry")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "acc_forw_asym", format="svg")


regressor = LinearRegression()
regressor.fit(np.array(temporal_factor_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
slope = regressor.coef_[0][0]
intercept = regressor.intercept_[0]
score = regressor.score(np.array(temporal_factor_list).reshape(-1, 1), np.array(accuracy_list).reshape(-1, 1))
if slope*np.min(temporal_factor_list)+intercept > np.min(accuracy_list):
    fit_line_point1 = [np.min(temporal_factor_list), slope*np.min(temporal_factor_list)+intercept]
else:
    fit_line_point1 = [(np.min(accuracy_list)-intercept)/slope, np.min(accuracy_list)]
if slope*np.max(temporal_factor_list)+intercept < np.max(accuracy_list):
    fit_line_point2 = [np.max(temporal_factor_list), slope*np.max(temporal_factor_list)+intercept]
else:
    fit_line_point2 = [(np.max(accuracy_list)-intercept)/slope, np.max(accuracy_list)]

plt.figure(figsize=(4, 3.3), dpi=180)
plt.scatter(temporal_factor_list, accuracy_list, alpha=0.5, color='tab:green')
plt.plot([fit_line_point1[0], fit_line_point2[0]], [fit_line_point1[1], fit_line_point2[1]], color='k', linestyle='--')
plt.text((np.max(temporal_factor_list)+np.min(temporal_factor_list))/2, 0.98, "R2={:.2f}".format(score), fontsize=11)
plt.ylabel("task accuracy")
plt.xlabel("temporal factor")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "acc_temporal_factor", format="svg")


# recall probability multiple models
exp_names = ["setup_gru_negmementreg_gamma-8", "setup_gru_negmementreg_gamma01-1",
             "setup_gru_negmementreg_gamma01-3", "setup_gru_negmementreg_gamma01-5"]
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

plt.figure(figsize=(4, 3.3), dpi=180)
for i in range(len(exp_names)):
    plt.scatter(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c='w', marker="o", edgecolors=ColorConverter().to_rgba('k', alpha=1-i*0.25), zorder=2)
    plt.plot(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c='k', alpha=1-i*0.25, label="seed {}".format(i+1), zorder=1)
    plt.scatter(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c='w', marker="o", edgecolors=ColorConverter().to_rgba('k', alpha=1-i*0.25), zorder=2)
    plt.plot(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c='k', alpha=1-i*0.25, zorder=1)
    # plt.scatter(np.array([0]), recall_probs[i][nsteps], c='r', marker="o", alpha=1-i*0.2)
plt.xlabel("lag")
plt.ylabel("conditional recall probability")
    # title = title if title else "conditional recall probability"
    # plt.title(title)

plt.legend(fontsize=11)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig("./figures", "crp", format="svg")



""" LCA tests """
# s = torch.ones(8) / 8.0
# s = torch.tensor([3.0, 2.8, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0])
# s = torch.tensor([0.09636872,0.11586493,0.12098483,0.13926123,0.13166645,0.11631244,0.15019856,0.1293428])
# s = (s - torch.mean(s)) / torch.std(s)
# print('original vector:\n', s)
# print('softmax vector (beta=0.1):\n', softmax(s.reshape(1, -1), beta=0.1))
# print('softmax vector (beta=0.01):\n', softmax(s.reshape(1, -1), beta=0.01))
# s = softmax(s.reshape(1, -1), beta=0.1)
# s = s.repeat(8, 1)
# lca = LCA(8, input_weight=1.0, lateral_inhibition=0.8)
# s_out = lca(s)

# for i in range(s_out.shape[1]):
#     plt.plot(s_out[:, i])
# plt.xlabel("time")
# plt.ylabel("value")
# plt.savefig("lca.png")

# print()
# print('after LCA:\n', s_out[-1])
# print('then after normalization:\n', s_out[-1].reshape(1, -1) / torch.sum(s_out[-1]))
# print('then after softmax (beta=0.2):\n', softmax(s_out[-1].reshape(1, -1) / torch.sum(s_out[-1]), beta=0.2))

# acc = [0.9416, 0.8013, 0.9893, 0.7351, 0.7825, 0.8213, 0.9256, 0.889, 0.8738, 0.9779,
#        0.8229, 0.9424, 0.9303, 0.9311, 0.891, 0.8073, 0.9083, 0.8925, 0.9463, 0.9043]
# forw_asym = [0.28, -0.01, 0.64, 0.02, 0.04, 0.02, 0.025, 0.023, 0.033, 0.62, 
#              0.077, 0.35, 0.77, 0.42, 0.031, 0.029, 0.28, 0.017, 0.013, -0.012]

# plt.figure(figsize=(4, 3))
# plt.scatter(acc, forw_asym)
# plt.xlabel("accuracy")
# plt.ylabel("forward asymmetry")
# plt.tight_layout()
# plt.savefig("acc_forw_asym.png")


""" Conditional EM Recall tests """
# env = ConditionalEMRecall(include_question_during_encode=True, has_question=False)
# env = MetaLearningEnv(env)
# obs, info = env.reset()
# print('memory_sequence:', env.memory_sequence)
# print('question_type:', env.question_type)
# print('question_value:', env.question_value)
# print('correct_answers:', env.memory_sequence[env.correct_answers_index])

# gt = env.get_ground_truth()
# actions = np.random.choice(gt, 9)
# correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions) 
# print('gt:', gt)
# print('actions:', actions)
# print('correct_actions:', correct_actions)
# print('wrong_actions:', wrong_actions)
# print('not_know_actions:', not_know_actions)

# actions = env.memory_sequence[env.correct_answers_index]
# actions_int = [0 for _ in range(8)]
# for action in actions:
#     actions_int.append(action[0]+action[1]*5)
#     actions_int.append(action[0]+action[1]*5)
# actions_int.append(26)
# actions_int.append(27)
# actions_int = np.array(actions_int)

# print(obs, info)
# cnt = 0
# while True:
#     # action = env.action_space.sample()
#     action = actions_int[cnt]
#     cnt += 1
#     print("action:", action, env.convert_action_to_stimuli(action))
#     obs, reward, done, info = env.step(action)
#     print(obs, reward, done, info)
#     if done:
#         break

