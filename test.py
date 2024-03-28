import os
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.memory.similarity.lca import LCA
from models.utils import softmax
from tasks import ConditionalEMRecall, MetaLearningEnv

from utils import load_dict

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


""" vary param for noise injection """
seq_len = [4,8,12,16]

setup_dir = Path("./experiments/RL/Noise/NBack/setups".format(seq_len))
setup_file = setup_dir / "setup.json"
setup = load_dict(setup_file)

for seq_len in [4, 8, 12, 16]:
    for noise in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        setup["model"]["flush_noise"] = noise
        setup["model"]["subclasses"][0]["capacity"] = seq_len
        for setup_train in setup["training"]:
            setup_train["env"]["memory_num"] = seq_len
        with open(setup_dir / "setup_seq{}_noise{}.json".format(seq_len, str(noise).replace(".", "")), "w") as f:
            json.dump(setup, f, indent=4)

""" change param for n-back task """

# setup_dir = Path("./experiments/RL/NBack/VarySeq/setups")
# files = os.listdir(setup_dir)
# for file in files:
#     with open(setup_dir / file, "r") as f:
#         setup = json.load(f)
#     setup["run_num"] = 50
#     setup["model"]["two_output"] = True
#     setup["training"][0]["trainer"]["sl_criterion"] = {
#         "class": "EncodingNBackCrossEntropyLoss",
#         "class_num": 51,
#         "nback": 1
#     }
#     setup["training"][1]["trainer"]["sl_criterion"] = {
#         "class": "EncodingNBackCrossEntropyLoss",
#         "class_num": 51,
#         "nback": 1
#     }
#     with open(setup_dir / file, "w") as f:
#         json.dump(setup, f, indent=4)

