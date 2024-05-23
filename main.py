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
from train import plot_accuracy_and_error, record_model


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
    for i in run_nums:
        print("run {}".format(i))
        general_setup = deepcopy(setup_origin)

        # get run name from setup name if not specified in setup dict
        run_name = general_setup.get("run_name", setup_name.split(".")[0])
        general_setup["run_name"] = run_name

        # parse setup
        # construct the model, environment, optimizer, etc.
        model_instances = parse_setup(general_setup, device)

        for run_name, model_instance in model_instances.items():
            run_name_with_num = run_name + "-{}".format(i)
            print("run_name: {}".format(run_name_with_num))

            # unpack model_instance
            model, model_for_record, envs, optimizers, schedulers, criterions, sl_criterions, training_setups, setup = model_instance

            # set up save model path
            model_save_path = exp_path/exp_dir/consts.SAVE_MODEL_FOLDER/setup["model_name"]/run_name_with_num
            model_save_path.mkdir(parents=True, exist_ok=True)

            # load trained model when not specified to train again
            # if load_model_path in setup is specified, may load from a different model (different experiment setup)
            if setup.get("load_model_name", None) is not None:
                load_run_name = setup["load_model_name"]
                load_run_name_with_path = load_run_name + "-{}".format(i)
            else:
                load_run_name_with_path = run_name_with_num
            # set up load model path
                
            model_load_path = exp_path/exp_dir/consts.SAVE_MODEL_FOLDER/setup["model_name"]/load_run_name_with_path
            if (not train or setup.get("load_saved_model", False)) and os.path.exists(model_load_path/"model.pt"):
                if setup.get("load_saved_model", False):
                    print("load saved model from {}".format(load_run_name_with_path))
                model.load_state_dict(torch.load(model_load_path/"model.pt", map_location=torch.device('cpu')))

            model.to(device)

            # train the model with each training setup
            if train or not os.path.exists(model_load_path/"model.pt"):
                training_session = 1
                for env, optimizer, scheduler, criterion, sl_criterion, training_setup in \
                        zip(envs, optimizers, schedulers, criterions, sl_criterions, training_setups):
                    if env and optimizer and scheduler and criterion:
                        print("\ntraining session {}".format(training_session))
                        training_func = training_setup["trainer"].pop("training_function", "supervised_train_model")
                        accuracies, errors = import_attr("train.{}".format(training_func))(model, env, optimizer, scheduler, setup, criterion, sl_criterion,
                            device=device, model_save_path=model_save_path, **training_setup["trainer"])
                        # save accuracy and error to file
                        np.save(model_save_path/"accuracy_{}.npy".format(training_session), np.array(accuracies))
                        np.save(model_save_path/"error_{}.npy".format(training_session), np.array(errors))
                        plot_accuracy_and_error(accuracies, errors, model_save_path, filename="accuracy_session_{}.png".format(training_session))
                        training_session += 1

            # record data of the model
            env = envs[-1]
            training_setup = training_setups[-1]["trainer"]
            if training_setup.get("reset_memory", True):
                print("reset memory during recording")
            else:
                print("not reset memory during recording")
            record_env = training_setup.get("record_env", [0])
            used_output = training_setup.get("used_output_index", [0])
            print("used_output:", used_output)
            assert len(record_env) == len(used_output)
            if env:
                if model_for_record is not None:
                    print("use record model setup")
                    model = model_for_record
                    model.load_state_dict(torch.load(model_load_path/"model.pt"))
                data_all_env = []
                for i in record_env:
                    data = record_model(model, env[i], used_output=used_output[i], 
                                        reset_memory=training_setup.get("reset_memory", True), 
                                        device=device, context_num=setup.get("context_num", 20))
                    data_all_env.append(data)

                model_all[run_name_with_num] = model
                data_all[run_name_with_num] = data_all_env

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

        exp_name = setup_name.split(".")[0]
        run_exp(data_all, model_all, env, paths, exp_name)


if __name__ == "__main__":
    _experiment, _setup_name, _device, _train, _debug, _test_accu, run_num, exp_file_name, _unknown_args = parse_args()
    main(_experiment, _setup_name, _device, _train, _debug, _test_accu, run_num, exp_file_name, _unknown_args)
