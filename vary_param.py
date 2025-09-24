import json
from pathlib import Path
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

from models.memory.similarity.lca import LCA
from models.utils import softmax
from tasks import FreeRecall2

from utils import load_dict, savefig


def main():

    """ Vary parameters """
    seq_len_all = [4,8,12,16]
    noise_all = [0, 0.2, 0.4, 0.6, 0.8, 1]
    gamma_all = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    gamma_all_actual = [0, 0.2, 0.4, 0.6, 0.8, 0.99]
    eta_all = [0.005, 0.01, 0.02, 0.04]
    wm_noise_all = [0, 0.01, 0.04, 0.09, 0.16, 0.25]

    # setup_dir = Path("./experiments/FreeRecall/VaryNoise/setups")
    # setup_dir = Path("./experiments/VaryAllSeq8NoNoise/setups")
    setup_dir = Path("./experiments/VarySeqLenNoise/setups")
    # setup_file = setup_dir / "setup.json"
    # setup = load_dict(setup_file)

    # for gamma in gamma_all:
    #     for eta in eta_all:
    #         setup["training"][-1]["trainer"]["criterion"]["criteria"][0]["eta"] = eta
    #         setup["training"][-1]["trainer"]["criterion"]["criteria"][0]["gamma"] = gamma
    #         with open(setup_dir / "setup_eta{}_gamma{}.json".format(str(eta).replace(".",""), str(gamma).replace(".", "")), "w") as f:
    #             json.dump(setup, f, indent=4)

    # for gamma in gamma_all:
    #     setup["training"][-1]["trainer"]["criterion"]["criteria"][0]["gamma"] = gamma
    #     with open(setup_dir / "setup_gamma{}.json".format(str(gamma).replace(".", "")), "w") as f:
    #         json.dump(setup, f, indent=4)

    # for eta in eta_all:
    #     setup["training"][-1]["trainer"]["criterion"]["criteria"][0]["eta"] = eta
    #     setup["training"][-1]["trainer"]["criterion"]["criteria"][0]["gamma"] = 0.999
    #     with open(setup_dir / "setup_pretrain_eta{}_gamma10.json".format(str(eta).replace(".","")), "w") as f:
    #         json.dump(setup, f, indent=4)

    # # for seq_len in seq_len_all:
    # for noise in noise_all:
    #     setup["model"]["flush_noise"] = noise
    #     setup["model_for_record"]["flush_noise"] = noise
    #     # setup["model"]["subclasses"][0]["capacity"] = seq_len
    #     # for i in range(len(setup["training"])):
    #     #     setup["training"][i]["env"][0]["memory_num"] = seq_len
    #     #     setup["training"][i]["env"][0]["retrieve_time_limit"] = seq_len
    #     with open(setup_dir / "setup_noise{}.json".format(str(noise).replace(".", "")), "w") as f:
    #         json.dump(setup, f, indent=4)

    # for gamma, gamma_actual in zip(gamma_all, gamma_all_actual):
    #     for noise in noise_all:
    #         for i in range(len(setup["training"])):
    #             setup["training"][i]["trainer"]["criterion"]["criteria"][0]["gamma"] = gamma_actual
    #         setup["model"]["flush_noise"] = noise
    #         setup["model_for_record"]["flush_noise"] = noise
    #         with open(setup_dir / "setup_gamma{}_noise{}.json".format(str(gamma).replace(".", ""), str(noise).replace(".", "")), "w") as f:
    #             json.dump(setup, f, indent=4)

    for seq_len in [16]:
        setup_file = setup_dir / "setup_seq{}.json".format(seq_len)
        setup = load_dict(setup_file)
        for gamma, gamma_actual in zip([0, 1], [0, 0.999]):
            for wm_noise in [0, 0.05, 0.1]:
                for i in range(len(setup["training"])):
                    setup["training"][i]["trainer"]["criterion"]["criteria"][0]["gamma"] = gamma_actual
                setup["model"]["wm_noise_prop"] = wm_noise
                setup["model_for_record"]["wm_noise_prop"] = wm_noise
                with open(setup_dir / "setup_seq{}_gamma{}_wmnoise{}.json".format(seq_len, str(gamma).replace(".", ""), str(wm_noise).replace(".", "")), "w") as f:
                    json.dump(setup, f, indent=4)

    # for wm_noise in wm_noise_all:
    #     setup["model"]["wm_noise_prop"] = wm_noise
    #     setup["model_for_record"]["wm_noise_prop"] = wm_noise
    #     with open(setup_dir / "setup_wmnoise{}.json".format(str(wm_noise).replace(".", "")), "w") as f:
    #         json.dump(setup, f, indent=4)


if __name__ == '__main__':
    main()
    
