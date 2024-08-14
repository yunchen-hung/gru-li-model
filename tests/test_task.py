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


""" Conditional EM task tests """
# env = ConditionalEMRecall(num_features=4, feature_dim=2, sequence_len=4, question_space=["choice"],
#                           include_question_during_encode=True, sum_feature_placeholder=True)
# env = MetaLearningEnv(env)
# env = PlaceHolderWrapper(env, 4)
# obs, info = env.reset()
# env.render()
# print(obs)
# for i in range(4):
#     action = 0
#     obs, reward, terminated, info = env.step(action)
#     print(action)
#     print(obs, reward, terminated, info)
# while not terminated:
#     action = 0
#     obs, reward, terminated, info = env.step(action)
#     print(action)
#     print(obs, reward, terminated, info)
# print()
# print(obs.shape)


""" Conditional Question Answer tests """
# seqlen = 4
# env = ConditionalQuestionAnswer(num_features=4, feature_dim=2, sequence_len=seqlen, 
#     include_question_during_encode=True)
# env = PlaceHolderWrapper(env, 11)
# env = MetaLearningEnv(env)

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
# print()
# print(obs.shape)

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



