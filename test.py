import csv
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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


""" plot temporal discount factor & accuracy - forward asymmetry & temporal factor """
exp_names = []
temporal_discount_factors = []
exp_names.append("setup_gru_negmementreg")
temporal_discount_factors.append(0.0)
for i in range(1, 9):
    exp_names.append("setup_gru_negmementreg_gamma0{}".format(i))
    temporal_discount_factors.append(float(i)/10)
exp_names.append("setup_gru_negmementreg_gamma")
temporal_discount_factors.append(1.0)

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
plt.figure(figsize=(5, 4.2), dpi=180)
plt.errorbar(temporal_discount_factors, mean_forward_asymmetry, yerr=std_forward_asymmetry, fmt='o', capsize=3)
plt.xlabel("temporal discount factor")
plt.ylabel("forward asymmetry")
plt.tight_layout()
savefig("./figures", "tdf_forw_asym", format="svg")

plt.figure(figsize=(5, 4.2), dpi=180)
plt.errorbar(temporal_discount_factors, mean_temporal_factor, yerr=std_temporal_factor, fmt='o', capsize=3)
plt.xlabel("temporal discount factor")
plt.ylabel("temporal factor")
plt.tight_layout()
savefig("./figures", "tdf_temporal_factor", format="svg")


# accuracy - forward asymmetry & temporal factor
regressor = LinearRegression()
regressor.fit(np.array(accuracy_list).reshape(-1, 1), np.array(forward_asymmetry_list).reshape(-1, 1))
slope = regressor.coef_[0][0]
intercept = regressor.intercept_[0]
score = regressor.score(np.array(accuracy_list).reshape(-1, 1), np.array(forward_asymmetry_list).reshape(-1, 1))
if slope*np.min(accuracy_list)+intercept > np.min(forward_asymmetry_list):
    fit_line_point1 = [np.min(accuracy_list)*1.05, (slope*np.min(accuracy_list)+intercept+0.05)*1.05]
else:
    fit_line_point1 = [(np.min(forward_asymmetry_list)-intercept)/slope*1.05, np.min(forward_asymmetry_list)*1.05]
if slope*np.max(accuracy_list)+intercept < np.max(forward_asymmetry_list):
    fit_line_point2 = [np.max(accuracy_list)*1.05, (slope*np.max(accuracy_list)+intercept)*1.05]
else:
    fit_line_point2 = [(np.max(forward_asymmetry_list)-intercept)/slope*1.05, np.max(forward_asymmetry_list)*1.05]

plt.figure(figsize=(5, 4.2), dpi=180)
plt.scatter(accuracy_list, forward_asymmetry_list, alpha=0.7)
plt.plot([fit_line_point1[0], fit_line_point2[0]], [fit_line_point1[1], fit_line_point2[1]], color='r', linestyle='-')
plt.text(0.8, np.max(forward_asymmetry_list), "R2={:.2f}".format(score))
plt.xlabel("task accuracy")
plt.ylabel("forward asymmetry")
plt.tight_layout()
savefig("./figures", "acc_forw_asym", format="svg")

regressor = LinearRegression()
regressor.fit(np.array(accuracy_list).reshape(-1, 1), np.array(temporal_factor_list).reshape(-1, 1))
slope = regressor.coef_[0][0]
intercept = regressor.intercept_[0]
score = regressor.score(np.array(accuracy_list).reshape(-1, 1), np.array(temporal_factor_list).reshape(-1, 1))
if slope*np.min(accuracy_list)+intercept > np.min(temporal_factor_list):
    fit_line_point1 = [np.min(accuracy_list), slope*np.min(accuracy_list)+intercept]
else:
    fit_line_point1 = [(np.min(temporal_factor_list)-intercept)/slope, np.min(temporal_factor_list)]
if slope*np.max(accuracy_list)+intercept < np.max(temporal_factor_list):
    fit_line_point2 = [np.max(accuracy_list), slope*np.max(accuracy_list)+intercept]
else:
    fit_line_point2 = [(np.max(temporal_factor_list)-intercept)/slope, np.max(temporal_factor_list)]

plt.figure(figsize=(5, 4.2), dpi=180)
plt.scatter(accuracy_list, temporal_factor_list, alpha=0.7)
plt.plot([fit_line_point1[0], fit_line_point2[0]], [fit_line_point1[1], fit_line_point2[1]], color='r', linestyle='-')
plt.text(0.8, np.max(temporal_factor_list), "R2={:.2f}".format(score))
plt.xlabel("task accuracy")
plt.ylabel("temporal factor")
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

plt.figure(figsize=(5, 4.2), dpi=180)
for i in range(len(exp_names)):
    plt.scatter(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c='b', marker="o", alpha=1-i*0.2, zorder=2)
    plt.plot(np.arange(-nsteps, 0), recall_probs[i][:nsteps], c='k', alpha=1-i*0.2, label="model {}".format(i+1), zorder=1)
    plt.scatter(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c='b', marker="o", alpha=1-i*0.2, zorder=2)
    plt.plot(np.arange(1, nsteps+1), recall_probs[i][nsteps+1:], c='k', alpha=1-i*0.2, zorder=1)
    plt.scatter(np.array([0]), recall_probs[i][nsteps], c='r', marker="o", alpha=1-i*0.2)
plt.xlabel("item position")
plt.ylabel("conditional recall probability")
    # title = title if title else "conditional recall probability"
    # plt.title(title)

plt.legend()

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

