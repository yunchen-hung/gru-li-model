import argparse
import subprocess
from pathlib import Path
import torch

import consts
from utils import load_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="", help="experiment name")
    parser.add_argument("--setup", type=str, default="setup.json", help="setup file name")
    parser.add_argument("-train", action='store_true', help="train the model from beginning, ignore the stored models")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args, unknown_args = parser.parse_known_args()

    experiment = args.exp
    assert args.setup != ""
    # setup_name = Path(args.setup).stem
    setup_name = args.setup
    device = args.device
    train = args.train

    return experiment, setup_name, device, train, unknown_args


def write_sbatch_script(experiment, setup_name, device, train, setup):
    run_name = setup.get("run_name", setup_name.split(".")[0])
    save_dir = exp_dir/consts.LOG_FOLDER/setup["model"]["class"]/run_name

    # parse run_num
    # if run_num is a int, number the runs with 1~run_num
    # if run_num is a list, number the runs with the numbers in the list
    # there could be int and list in the list, e.g. [1, [2, 4], 5] means run 1, 2, 3, 4, 5
    run_num = setup.get("run_num", 1)
    if isinstance(run_num, list):
        run_nums = []
        for i in range(len(run_num)):
            if isinstance(run_num[i], int):
                run_nums.append(run_num[i])
            elif isinstance(run_num[i], list):
                assert len(run_num[i]) == 2
                run_nums.extend(list(range(run_num[i][0], run_num[i][1]+1)))
    else:
        run_nums = list(range(1, run_num+1))

    shell_path = save_dir / "run.sh"
    shell_file = open(shell_path, "w")
    shell_file.write(
        ""
    )




if __name__ == "__main__":
    experiment, setup_name, device, train, unknown_args = parse_args()

    exp_dir = Path("{}/{}".format(consts.EXPERIMENT_FOLDER, experiment).replace(".", "/"))
    setup = load_dict(exp_dir/consts.SETUP_FOLDER/setup_name)

    write_sbatch_script(experiment, setup_name, device, train, setup)
