import os
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.memory.similarity.lca import LCA
from models.utils import softmax
from tasks import ConditionalEMRecall, MetaLearningEnv, ConditionalQuestionAnswer, FreeRecallRepeat, \
    FreeRecall, PlaceHolderWrapper
from analysis.behavior.temporal_factor import TemporalFactor

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

""" temporal factor tests """
# seqlen = 16
# tf = TemporalFactor()
# results = []
# for i in range(10):
#     memory_context = np.arange(1, seqlen+1).reshape(1, -1)
#     # actions = np.random.permutation(np.arange(1, seqlen+1)).reshape(1, -1)
#     actions = np.array([1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10, 8, 9]).reshape(1, -1)
#     # actions = np.arange(1, seqlen+1).reshape(1, -1)
#     # print(actions)
#     result = np.mean(tf.fit(memory_context, actions))
#     results.append(result)
# print(np.mean(results))


""" Conditional EM task tests """
env = ConditionalEMRecall(num_features=4, feature_dim=2, sequence_len=4, question_space=["choice"],
                          include_question_during_encode=True, sum_feature_placeholder=True)
env = MetaLearningEnv(env)
env = PlaceHolderWrapper(env, 4)
obs, info = env.reset()
env.render()
print(obs)
for i in range(4):
    action = 0
    obs, reward, terminated, info = env.step(action)
    print(action)
    print(obs, reward, terminated, info)
while not terminated:
    action = 0
    obs, reward, terminated, info = env.step(action)
    print(action)
    print(obs, reward, terminated, info)
print()
print(obs.shape)


""" Conditional Question Answer tests """
seqlen = 4
env = ConditionalQuestionAnswer(num_features=4, feature_dim=2, sequence_len=seqlen, 
    include_question_during_encode=True)
env = PlaceHolderWrapper(env, 11)
env = MetaLearningEnv(env)

obs, info = env.reset()
env.render()
print(obs)
for i in range(seqlen):
    action = 0
    obs, reward, terminated, info = env.step(action)
    print(action)
    print(obs, reward, terminated, info)
actions = [env.action_space.n-1]*(seqlen-1) + [0]
cnt = 0
while not terminated:
    action = actions[cnt]
    cnt += 1
    obs, reward, terminated, info = env.step(action)
    print(action)
    print(obs, reward, terminated, info)
print()
print(obs.shape)

# obs, info = env.reset()
# env.render()
# print(obs)
# for i in range(seqlen):
#     action = 0
#     obs, reward, terminated, info = env.step(action)
#     print(action)
#     print(obs, reward, terminated, info)
# actions = [env.action_space.n-1]*(seqlen-1) + [0]
# cnt = 0
# while not terminated:
#     action = actions[cnt]
#     cnt += 1
#     obs, reward, terminated, info = env.step(action)
#     print(action)
#     print(obs, reward, terminated, info)

# answer = np.zeros(2)
# # prev_answer = np.zeros(2)
# cnts = np.zeros(seqlen+1)
# for i in range(10000):
#     env.reset()
#     if env.answer is not None:
#         answer[env.answer] += 1
#     # if np.array_equal(env.answer, prev_answer):
#     #     print(env.answer)
#     # prev_answer = env.answer
#     cnts[env.cnt] += 1
# print(answer)
# print(cnts)


""" vary param for noise injection """
# seq_len = [4,8,12,16]

# setup_dir = Path("./experiments/RL/Noise/NBack0/setups")
# setup_file = setup_dir / "setup.json"
# setup = load_dict(setup_file)

# for seq_len in [4, 8, 12, 16]:
#     for noise in [0, 0.2, 0.4, 0.6, 0.8, 1]:
#         setup["model"]["flush_noise"] = noise
#         setup["model"]["subclasses"][0]["capacity"] = seq_len
#         for setup_train in setup["training"]:
#             setup_train["env"]["memory_num"] = seq_len
#         setup["run_num"] = 20
#         with open(setup_dir / "setup_seq{}_noise{}.json".format(seq_len, str(noise).replace(".", "")), "w") as f:
#             json.dump(setup, f, indent=4)

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


""" test repeat free recall task """
# env = FreeRecallRepeat()
# # print(env.memory_sequence)
# obs, info = env.reset()
# memory_seq = env.memory_sequence
# print(memory_seq)
# done = False
# cnt = 0
# while not done:
#     if cnt < env.memory_num:
#         action = torch.tensor([memory_seq[0, cnt]])
#     else:
#         action = torch.tensor([memory_seq[0, cnt-5]])
#     obs, reward, done, info = env.step(action)
#     print(action)
#     print(obs, reward, done, info)
#     cnt += 1

