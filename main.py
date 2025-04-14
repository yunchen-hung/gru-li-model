from copy import deepcopy
from sys import platform
import os
import argparse
from pathlib import Path
import ast
import numpy as np
import torch

import consts
from utils import load_setup, parse_setup, import_attr
from train.utils import save_model
from train import plot_accuracy_and_error, record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="", help="experiment name")
    parser.add_argument("--setup", type=str, default="setup.json", help="setup file name")
    parser.add_argument("-train", action='store_true', help="train the model from beginning, ignore the stored models")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("-debug", action='store_true', help="debug mode, don't save results")
    parser.add_argument("-test_accu", action='store_true', help="test accuracy when loading models")
    parser.add_argument("--run_num", default=None, help="number of runs, can be list or int")
    parser.add_argument("--exp_file", type=str, default="experiment", help="experiment file name")

    args, unknown_args = parser.parse_known_args()

    experiment = args.exp
    assert args.setup != ""
    # setup_name = Path(args.setup).stem
    setup_name = args.setup
    device = args.device
    train = args.train
    debug = args.debug
    test_accu = args.test_accu
    run_num = args.run_num
    exp_file_name = args.exp_file
    if run_num is not None and isinstance(run_num, str) and run_num[0] == "[" and run_num[-1] == "]":
        run_num = ast.literal_eval(run_num)

    return experiment, setup_name, device, train, debug, test_accu, run_num, exp_file_name, unknown_args

def main(experiment, setup_name, device='cuda' if torch.cuda.is_available() else 'cpu', train=False, debug=False, test_accu=False, 
         run_num=None, exp_file_name="experiment", unknown_args=None):
    # load setup
    # exp_dir = Path("{}/{}".format(consts.EXPERIMENT_FOLDER, experiment).replace(".", "/"))
    exp_dir = Path(experiment.replace(".", "/"))
    setup_origin = load_setup(Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.SETUP_FOLDER/setup_name)

    # parse run_num
    # if run_num is a int, number the runs with 1~run_num
    # if run_num is a list, number the runs with the numbers in the list
    # there could be int and list in the list, e.g. [1, [2, 4], 5] means run 1, 2, 3, 4, 5
    run_num = setup_origin.get("run_num", 1) if run_num is None else run_num
    if isinstance(run_num, list):
        run_nums = []
        for i in range(len(run_num)):
            if isinstance(run_num[i], int):
                run_nums.append(run_num[i])
            elif isinstance(run_num[i], list):
                assert len(run_num[i]) == 2
                run_nums.extend(list(range(run_num[i][0], run_num[i][1]+1)))
    else:
        run_nums = list(range(0, run_num))

    print("device:", device)

    # build soft link for saved models and figures
    if platform == "linux":
        exp_path = Path(consts.CLUSTER_FOLDER)/exp_dir/consts.SAVE_MODEL_FOLDER
        if not os.path.exists(exp_path) or not os.path.exists(Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.SAVE_MODEL_FOLDER):
            os.makedirs(exp_path, exist_ok=True)
            os.symlink(exp_path, Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.SAVE_MODEL_FOLDER)
        figuire_path = Path(consts.CLUSTER_FOLDER)/exp_dir/consts.FIGURE_FOLDER
        if not os.path.exists(figuire_path) or not os.path.exists(Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.FIGURE_FOLDER):
            os.makedirs(figuire_path, exist_ok=True)
            os.symlink(figuire_path, Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.FIGURE_FOLDER)
    else:
        exp_path = Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.SAVE_MODEL_FOLDER
        figuire_path = Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.FIGURE_FOLDER

    model_all, data_all = {}, {}
    checkpoints_all = {}
    for i in run_nums:
        print("run {}".format(i))
        general_setup = deepcopy(setup_origin)

        # get run name from setup name if not specified in setup dict
        run_name = general_setup.get("run_name", setup_name.split(".")[0])
        general_setup["run_name"] = run_name

        # parse setup
        # construct the model, environment, optimizer, etc.
        model_instances = parse_setup(general_setup, device)
        # print(model_instances)

        for run_name, model_instance in model_instances.items():
            run_name_with_num = run_name + "-{}".format(i)
            print("run_name: {}".format(run_name_with_num))

            # unpack model_instance
            model, model_for_record, envs, single_env, optimizers, schedulers, criterions, sl_criterions, ax_criterions, training_setups, setups = model_instance

            # set up save model path
            model_save_path = exp_path/setups[0]["model_name"]/run_name_with_num
            model_save_path.mkdir(parents=True, exist_ok=True)

            # load trained model when not specified to train again
            # if load_model_path in setup is specified, may load from a different model (different experiment setup)
            if setups[0].get("load_model_name", None) is not None:
                load_run_name = setups[0]["load_model_name"]
                load_run_name_with_path = load_run_name + "-{}".format(i)
            else:
                load_run_name_with_path = run_name_with_num
            # set up load model path
                
            model_load_path = exp_path/setups[0]["model_name"]/load_run_name_with_path
            if (not train or setups[0].get("load_saved_model", False)) and os.path.exists(model_load_path/"model.pt"):
                if setups[0].get("load_saved_model", False):
                    print("load saved model from {}".format(load_run_name_with_path))
                model.load_state_dict(torch.load(model_load_path/"model.pt", map_location=torch.device('cpu'), weights_only=True))
            # print(exp_path, setup["model_name"], load_run_name_with_path)

            model.to(device)

            # train the model with each training setup
            save_model(model, model_save_path, filename="0.pt")
            if train or not os.path.exists(model_load_path/"model.pt"):
                training_session = 1
                for env, optimizer, scheduler, criterion, sl_criterion, ax_criterion, training_setup, setup in \
                        zip(envs, optimizers, schedulers, criterions, sl_criterions, ax_criterions, training_setups, setups):
                    if env and optimizer and scheduler and (criterion or sl_criterion):
                        print("\ntraining session {}".format(training_session))
                        training_func = training_setup["trainer"].pop("training_function", "train")

                        # model = torch.compile(model)
                        # model = torch.compile(model, mode="reduce-overhead")
                        # criterion = torch.compile(criterion)

                        # accuracies, errors = import_attr("train.{}".format(training_func))(model, env, optimizer, scheduler, setup, criterion, sl_criterion,
                        #     device=device, model_save_path=model_save_path, **training_setup["trainer"])
                        accuracies, errors = import_attr("train.{}".format(training_func))(setup, model, env, optimizer, scheduler, criterion, sl_criterion,
                            ax_criterion, device=device, model_save_path=model_save_path, session_num=training_session, **training_setup["trainer"])
                        # save accuracy and error to file
                        np.save(model_save_path/"accuracy_{}.npy".format(training_session), np.array(accuracies))
                        np.save(model_save_path/"error_{}.npy".format(training_session), np.array(errors))
                        plot_accuracy_and_error(accuracies, errors, model_save_path, filename="accuracy_session_{}.png".format(training_session))
                        training_session += 1

            # record data of the model
            # env = envs[-1]
            env = single_env
            training_setup = training_setups[-1]["trainer"]
            if training_setup.get("reset_memory", True):
                print("reset memory during recording")
            else:
                print("not reset memory during recording")
            record_env = setups[-1].get("record_env", [0])
            used_output = setups[-1].get("used_output_index", [0])
            print("used_output:", used_output)
            assert len(record_env) == len(used_output)
            if env:
                if model_for_record is not None:
                    print("use record model setup")
                    model = model_for_record
                    model.load_state_dict(torch.load(model_load_path/"model.pt", map_location=torch.device('cpu'), weights_only=True))
                data_all_env = []
                for i in record_env:
                    data = record(model, env[i], used_output=used_output[i], 
                                        reset_memory=training_setup.get("reset_memory", True), 
                                        device=device, context_num=setups[-1].get("context_num", 20))
                    data_all_env.append(data)

                model_all[run_name_with_num] = model
                data_all[run_name_with_num] = data_all_env

            # load checkpoints
            if "load_checkpoints" in setup_origin and setup_origin["load_checkpoints"]:
                checkpoints = []
                checkpoint_epoch_nums = []
                checkpoint_session_nums = []
                # get all files in model_load_path with the pattern "*.pt" and not "model.pt"
                checkpoint_files = list(model_load_path.glob("*.pt"))
                # Extract checkpoint numbers and sort them
                checkpoint_numbers = []
                for file in checkpoint_files:
                    if file.name != "model.pt":
                        try:
                            num = int(file.stem.split('.')[0])  # Get number after last underscore
                            checkpoint_numbers.append(num)
                        except ValueError:
                            continue
                checkpoint_numbers.sort()
                
                # Reorder checkpoint files based on sorted numbers
                checkpoint_files = [model_load_path/f"{num}.pt" for num in checkpoint_numbers]
                for checkpoint_file in checkpoint_files:
                    if checkpoint_file.name != "model.pt":
                        # print(checkpoint_file.name)
                        checkpoints.append(torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=True))
                        checkpoint_epoch_nums.append(int(checkpoint_file.stem.split('.')[0].split('_')[1]))
                        checkpoint_session_nums.append(int(checkpoint_file.stem.split('.')[0].split('_')[0]))
                print(checkpoint_session_nums, checkpoint_epoch_nums)
                checkpoints_all[run_name_with_num] = [checkpoint_session_nums, checkpoint_epoch_nums, checkpoints]
            else:
                checkpoints_all = None

    # run experiment
    run_exp = import_attr("{}.{}.{}.run".format(consts.EXPERIMENT_FOLDER.replace('/', '.'), experiment, exp_file_name))
    
    if env:
        # record_env = setup.get("record_env", [0])
        # used_output = setup.get("used_output", [0])
        # for i in record_env:
        #     env_name = env[i].__class__.__name__
        #     if exp_file_name == "experiment":
        #         paths = {"fig": figuire_path/setup_origin["model"]["class"]}
        #     else:
        #         paths = {"fig": figuire_path/exp_file_name/setup_origin["model"]["class"]}
        #     if len(record_env) > 1:
        #         paths['fig'] = paths['fig']/env_name

        #     exp_name = setup_name.split(".")[0]
        #     run_exp(data_all, model_all, env[i], paths, exp_name)
        if exp_file_name == "experiment":
            paths = {"fig": figuire_path/setup_origin["model"]["class"]}
        else:
            paths = {"fig": figuire_path/exp_file_name/setup_origin["model"]["class"]}

        kwargs = {
            "criterion": criterions[-1]
        }

        exp_name = setup_name.split(".")[0]
        run_exp(data_all, model_all, env, paths, exp_name, checkpoints=checkpoints_all, **kwargs)


if __name__ == "__main__":
    _experiment, _setup_name, _device, _train, _debug, _test_accu, run_num, exp_file_name, _unknown_args = parse_args()
    main(_experiment, _setup_name, _device, _train, _debug, _test_accu, run_num, exp_file_name, _unknown_args)
