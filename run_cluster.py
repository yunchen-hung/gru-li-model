import os
import argparse
import inspect
import subprocess
from pathlib import Path
import torch

import consts
from utils import load_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="", help="experiment name")
    parser.add_argument("--setup", type=str, default="setup.json", help="setup file name")
    parser.add_argument("--time", type=int, default=10, help="time limit for each run (hours)")
    parser.add_argument("-train", action='store_true', help="train the model from beginning, ignore the stored models")
    parser.add_argument("--cpus_per_task", type=int, default=1, help="number of cpus per task")
    parser.add_argument("--exp_file", type=str, default="experiment", help="experiment file name")
    parser.add_argument("--mem", type=int, default=32, help="memory limit for each run (GB)")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args, unknown_args = parser.parse_known_args()

    experiment = args.exp
    assert args.setup != ""
    # setup_name = Path(args.setup).stem
    setup_name = args.setup
    time_limit = args.time
    device = args.device
    train = args.train
    cpus_per_task = args.cpus_per_task
    exp_file_name = args.exp_file
    mem = args.mem

    return experiment, setup_name, time_limit, device, train, cpus_per_task, exp_file_name, mem, unknown_args


def write_sbatch_script(experiment, setup_name, exp_dir, device, train, cpus_per_task, setup, time_limit=5, exp_file_name="experiment", mem=32):
    run_name = setup.get("run_name", setup_name.split(".")[0])
    os.makedirs(exp_dir/setup["model"]["class"]/run_name, exist_ok=True)
    save_dir = exp_dir/setup["model"]["class"]/run_name

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

    # create folder for stdout and stderr
    os.makedirs(save_dir/"stdout", exist_ok=True)
    os.makedirs(save_dir/"stderr", exist_ok=True)

    shell_path = save_dir / "run.sh"
    shell_file = open(shell_path, "w")
    shell_file.write(
        f"#!/bin/bash\n" +
        f"#SBATCH --job-name={experiment}.{run_name}\n" +
        # f"#SBATCH --cpus-per-task=1\n" +
        f"#SBATCH --time={time_limit}:00:00\n" +
        f'#SBATCH --cpus-per-task={cpus_per_task}\n' +
        # f"#SBATCH --mem-per-cpu=16G\n" +
        f"#SBATCH --mem={mem}G\n" +
        f"#SBATCH -e {save_dir}/stderr/slurm-%A_%a.err\n" +
        f"#SBATCH -o {save_dir}/stdout/slurm-%A_%a.out\n" +
        f"#SBATCH --array=0-{len(run_nums)-1}\n" + 
        f"\n" +

        inspect.cleandoc(f"python -u main.py --exp {experiment} --setup {setup_name} --run_num \"[$SLURM_ARRAY_TASK_ID]\" --device {device} --exp_file {exp_file_name}")
    )
    if train:
        shell_file.write(" -train")
    shell_file.close()
    return shell_path


def run_cluster(experiment, setup_name, time_limit, device, train, cpus_per_task, exp_file_name, mem):
    exp_dir = Path("{}/{}".format(consts.CLUSTER_FOLDER, experiment).replace(".", "/")) / consts.LOG_FOLDER
    link_dir = Path("{}/{}".format(consts.EXPERIMENT_FOLDER, experiment).replace(".", "/")) / consts.LOG_FOLDER
    if not os.path.exists(exp_dir) or not os.path.exists(link_dir):
        os.makedirs(exp_dir, exist_ok=True)
        os.symlink(exp_dir, link_dir)
    setup = load_dict(Path("{}/{}".format(consts.EXPERIMENT_FOLDER, experiment).replace(".", "/"))/consts.SETUP_FOLDER/setup_name)

    shell_path = write_sbatch_script(experiment, setup_name, exp_dir, device, train, cpus_per_task, setup, time_limit, exp_file_name, mem)

    subprocess.run(f"sbatch {shell_path}", shell=True)


if __name__ == "__main__":
    experiment, setup_name, time_limit, device, train, cpus_per_task, exp_file_name, mem, unknown_args = parse_args()

    run_cluster(experiment, setup_name, time_limit, device, train, cpus_per_task, exp_file_name, mem)
