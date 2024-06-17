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

