import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from models.memory.similarity.lca import LCA
from models.utils import softmax
from tasks import ConditionalEMRecall , MetaLearningEnv, ConditionalQuestionAnswer, FreeRecallRepeat, \
    FreeRecall, PlaceHolderWrapper

from utils import load_dict


def main():
    def make_env(seed):
        env = FreeRecall()
        env.seed(seed)
        return env

    seeds = np.random.randint(0, 100000, 3)

    setup = {"vocabulary_num": 15}

    env = gym.vector.SyncVectorEnv([
        # lambda: MetaLearningEnv(ConditionalQuestionAnswer(seed=seeds[i], num_features=3, feature_dim=2, sequence_len=4))
        lambda: FreeRecall(seed=seeds[i], **setup)
        # make_env(seeds[i])
        for i in range(3)
    ])

    print(env.num_envs)

    for i in range(3):
        obs, info = env.reset()
        print(obs)
        # terminated = np.array([False] * 3)
        # cnt = 0
        # while not terminated.all():
        #     action = env.action_space.sample()
        #     cnt += 1
        #     obs, reward, terminated, _, info = env.step(action)
        #     print(action)
        #     print(obs, reward, terminated, info)
        #     # if cnt > 20:
        #     #     break
        # print(obs.shape)
        # print()
        


if __name__ == '__main__':
    main()
    
