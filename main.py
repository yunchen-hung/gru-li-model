from copy import deepcopy
from sys import platform
import os
import argparse
from pathlib import Path
import ast
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
    if run_num is not None and isinstance(run_num, str) and run_num[0] == "[" and run_num[-1] == "]":
        run_num = ast.literal_eval(run_num)

    return experiment, setup_name, device, train, debug, test_accu, run_num, unknown_args

def main(experiment, setup_name, device='cuda' if torch.cuda.is_available() else 'cpu', train=False, debug=False, test_accu=False, run_num=None, unknown_args=None):
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
            model, model_for_record, envs, optimizers, schedulers, criterions, training_setups, setup = model_instance

            # set up model save path
            if platform == "linux":
                model_save_path = Path(consts.CLUSTER_SAVE_MODEL_FOLDER)/exp_dir/setup["model_name"]/run_name_with_num
            else:
                model_save_path = Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.SAVE_MODEL_FOLDER/setup["model_name"]/run_name_with_num
            model_save_path.mkdir(parents=True, exist_ok=True)

            # load trained model when not specified to train again
            # if load_model_path in setup is specified, may load from a different model (different experiment setup)
            if setup.get("load_model_name", None) is not None:
                load_run_name = setup["load_model_name"]
                load_run_name_with_path = load_run_name + "-{}".format(i)
            else:
                load_run_name_with_path = run_name_with_num
            if platform == "linux":
                model_load_path = Path(consts.CLUSTER_SAVE_MODEL_FOLDER)/exp_dir/setup["model_name"]/load_run_name_with_path
            else:
                model_load_path = Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.SAVE_MODEL_FOLDER/setup["model_name"]/load_run_name_with_path
            if (not train or setup.get("load_saved_model", False)) and os.path.exists(model_load_path/"model.pt"):
                if setup.get("load_saved_model", False):
                    print("load saved model from {}".format(load_run_name_with_path))
                model.load_state_dict(torch.load(model_load_path/"model.pt", map_location=torch.device('cpu')))

            model.to(device)

            torch.autograd.set_detect_anomaly(True)

            # train the model with each training setup
            if train or not os.path.exists(model_load_path/"model.pt"):
                training_session = 1
                for env, optimizer, scheduler, criterion, training_setup in zip(envs, optimizers, schedulers, criterions, training_setups):
                    if env and optimizer and scheduler and criterion:
                        print("\ntraining session {}".format(training_session))
                        training_session += 1
                        training_func = training_setup["trainer"].pop("training_function", "supervised_train_model")
                        accuracies, errors = import_attr("train.{}".format(training_func))(model, env, optimizer, scheduler, setup, criterion, device=device, 
                            model_save_path=model_save_path, **training_setup["trainer"])
                    plot_accuracy_and_error(accuracies, errors, model_save_path)

            # record data of the model
            env = envs[-1]
            if env:
                if model_for_record is not None:
                    print("use record model setup")
                    model = model_for_record
                    model.load_state_dict(torch.load(model_load_path/"model.pt"))
                data = record_model(model, env, device=device, context_num=setup.get("context_num", 20))

            model_all[run_name_with_num] = model
            data_all[run_name_with_num] = data

    paths = {"fig": Path(consts.EXPERIMENT_FOLDER)/exp_dir/consts.FIGURE_FOLDER/setup_origin["model"]["class"]}

    # run experiment
    run_exp = import_attr("{}.{}.experiment.run".format(consts.EXPERIMENT_FOLDER.replace('/', '.'), experiment))
    if env:
        exp_name = setup_name.split(".")[0]
        run_exp(data_all, model_all, env, paths, exp_name)


if __name__ == "__main__":
    _experiment, _setup_name, _device, _train, _debug, _test_accu, run_num, _unknown_args = parse_args()
    main(_experiment, _setup_name, _device, _train, _debug, _test_accu, run_num, _unknown_args)
