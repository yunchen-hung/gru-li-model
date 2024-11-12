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


def main():
    # def make_env(seed):
    #     env = FreeRecall()
    #     env.seed(seed)
    #     return env

    # seeds = np.random.randint(0, 100000, 3)

    # setup = {"vocabulary_num": 15}

    # env = gym.vector.SyncVectorEnv([
    #     lambda: MetaLearningEnv(ConditionalQuestionAnswer(seed=seeds[i], num_features=3, feature_dim=2, sequence_len=4, no_early_stop=False))
    #     # lambda: FreeRecall(seed=seeds[i], **setup)
    #     # make_env(seeds[i])
    #     for i in range(3)
    # ])

    # print(env.num_envs)

    # for i in range(1):
    #     obs, info = env.reset()
    #     print(obs)
    #     terminated = np.array([False] * 3)
    #     cnt = 0
    #     while not terminated.all():
    #         print("timestep: ", cnt)
    #         action = env.action_space.sample()
    #         cnt += 1
    #         obs, reward, done, _, info = env.step(action)
    #         terminated = np.logical_or(terminated, done)
    #         print(action)
    #         print(obs, reward, terminated, info)
    #         # if cnt > 20:
    #         #     break
    #         print()
    #     print(obs.shape)
    #     print()

    """ Conditional Question Answer tests """
    # seqlen = 8
    # env = ConditionalQuestionAnswer(num_features=4, feature_dim=2, sequence_len=seqlen, 
    #     include_question_during_encode=True)
    # env = PlaceHolderWrapper(env, 11)
    # env = MetaLearningEnv(env)

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

    """ Vary parameters """
    # seq_len_all = [8,16]
    # noise_all = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # gamma_all = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # setup_dir = Path("./experiments/FreeRecall/VaryNoise/setups")
    # setup_file = setup_dir / "setup.json"
    # setup = load_dict(setup_file)

    # # for gamma in gamma_all:
    # #     setup["training"][-1]["trainer"]["criterion"]["criteria"][0]["gamma"] = gamma
    # #     with open(setup_dir / "setup_gamma{}.json".format(str(gamma).replace(".", "")), "w") as f:
    # #         json.dump(setup, f, indent=4)

    # for seq_len in seq_len_all:
    #     for noise in noise_all:
    #         setup["model"]["flush_noise"] = noise
    #         setup["model"]["subclasses"][0]["capacity"] = seq_len
    #         for i in range(len(setup["training"])):
    #             setup["training"][i]["env"][0]["memory_num"] = seq_len
    #             setup["training"][i]["env"][0]["retrieve_time_limit"] = seq_len
    #         with open(setup_dir / "setup_seq{}_noise{}.json".format(seq_len, str(noise).replace(".", "")), "w") as f:
    #             json.dump(setup, f, indent=4)



if __name__ == '__main__':
    main()
    
