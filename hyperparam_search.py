import os
import argparse
from pathlib import Path
import itertools
import torch

import consts
from utils import load_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="", help="experiment name")
    parser.add_argument("--setup", type=str, default="setup.json", help="setup file name")
    parser.add_argument("--time", type=int, default=8, help="time limit for each run")
    parser.add_argument("-train", action='store_true', help="train the model from beginning, ignore the stored models")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args, unknown_args = parser.parse_known_args()

    experiment = args.exp
    assert args.setup != ""
    # setup_name = Path(args.setup).stem
    setup_name = args.setup
    time_limit = args.time
    device = args.device
    train = args.train

    return experiment, setup_name, time_limit, device, train, unknown_args


def sample_params(experiment, setup_name):
    # load setup
    exp_dir = Path("{}/{}".format(consts.EXPERIMENT_FOLDER, experiment).replace(".", "/"))
    setup = load_dict(exp_dir/consts.SETUP_FOLDER/setup_name)

    # params to sample
    param_num = 10
    model_params = {
        "init_state_type": ["zeros", "train", "train_diff"],
        "hidden_dim": [64, 128, 256],
        "subclasses": {
            "memory_module": {
                "subclasses": {
                    "similarity_measure": {
                        "similarity_measure": ["cosine", "l1", "l2"],
                        "softmax_temperature": [0.01, 0.02, 0.05, 0.1],
                        "process_similarity": ["softmax", "normalize_softmax"]
                    }
                }
            }
        }
    }
    train1_params = {}
    train2_params = {
        "trainer": {
            "memory_reg_weight": [0.0, 1e-4, 1e-3, 1e-2],
            "step_iter": [1, 2, 4],
            "criterion": {
                "eta": [0.0, 1e-4, 1e-3, 1e-2]
            },
            "optimizer": {
                "lr": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
            }
        }
    }
    dependency = {
        "init_state_type": {
            "evolve_state_between_phases": [True, False, False]
        }
    }

    setups = []
    


if __name__ == "__main__":
    experiment, setup_name, time_limit, device, train, unknown_args = parse_args()

