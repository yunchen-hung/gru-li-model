from copy import deepcopy
import os
import argparse
from pathlib import Path
import torch

import consts
from utils import load_setup, parse_setup, import_attr
from train import plot_accuracy_and_error, record_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="", help="")
    parser.add_argument("--setup", type=str, default="setup.json", help="")
    parser.add_argument("-train", action='store_true', help="")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="")
    parser.add_argument("-debug", action='store_true', help="debug mode, don't save results")
    parser.add_argument("-test_accu", action='store_true', help="test accuracy when loading models")

    args, unknown_args = parser.parse_known_args()

    experiment = args.experiment
    assert args.setup != ""
    # setup_name = Path(args.setup).stem
    setup_name = args.setup
    device = args.device
    train = args.train
    debug = args.debug
    test_accu = args.test_accu

    return experiment, setup_name, device, train, debug, test_accu, unknown_args

def main(experiment, setup_name, device='cuda' if torch.cuda.is_available() else 'cpu', train=False, debug=False, test_accu=False, unknown_args=None):
    exp_dir = Path("{}/{}".format(consts.EXPERIMENT_FOLDER, experiment).replace(".", "/"))
    setup_origin = load_setup(exp_dir/consts.SETUP_FOLDER/setup_name)

    run_num = setup_origin.get("run_num", 1)
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

    model_all, data_all = {}, {}
    for i in run_nums:
        print("run {}".format(i))
        general_setup = deepcopy(setup_origin)

        run_name = general_setup.get("run_name", setup_name.split(".")[0])
        general_setup["run_name"] = run_name

        model_instances = parse_setup(general_setup, device)

        for run_name, model_instance in model_instances.items():
            run_name_with_num = run_name + "-{}".format(i)
            print("run_name: {}".format(run_name_with_num))

            model, model_for_record, envs, optimizers, schedulers, criterions, training_setups, setup = model_instance

            model_save_path = exp_dir/consts.SAVE_MODEL_FOLDER/setup["model_name"]/run_name_with_num
            model_save_path.mkdir(parents=True, exist_ok=True)

            if (not train or setup.get("load_saved_model", False)) and os.path.exists(model_save_path/"model.pt"):
                if setup.get("load_saved_model", False):
                    print("load saved model")
                model.load_state_dict(torch.load(model_save_path/"model.pt"))

            print("device: ", device)
            model.to(device)

            if train or not os.path.exists(model_save_path/"model.pt"):
                training_session = 1
                for env, optimizer, scheduler, criterion, training_setup in zip(envs, optimizers, schedulers, criterions, training_setups):
                    if env and optimizer and scheduler and criterion:
                        print("\ntraining session {}".format(training_session))
                        training_session += 1
                        training_func = training_setup["trainer"].pop("training_function", "supervised_train_model")
                        accuracies, errors = import_attr("train.{}".format(training_func))(model, env, optimizer, scheduler, setup, criterion, device=device, 
                            model_save_path=model_save_path, **training_setup["trainer"])
                    plot_accuracy_and_error(accuracies, errors, model_save_path)

            env = envs[-1]
            if env:
                if model_for_record is not None:
                    print("use record model setup")
                    model = model_for_record
                    model.load_state_dict(torch.load(model_save_path/"model.pt"))
                data = record_model(model, env, device=device, context_num=setup.get("context_num", 20))

            model_all[run_name_with_num] = model
            data_all[run_name_with_num] = data

    paths = {"fig": exp_dir/consts.FIGURE_FOLDER/setup_origin["model"]["class"]}

    run_exp = import_attr("{}.{}.experiment.run".format(consts.EXPERIMENT_FOLDER.replace('/', '.'), experiment))
    if env:
        exp_name = setup_name.split(".")[0]
        run_exp(data_all, model_all, env, paths, exp_name)


if __name__ == "__main__":
    _experiment, _setup_name, _device, _train, _debug, _test_accu, _unknown_args = parse_args()
    main(_experiment, _setup_name, _device, _train, _debug, _test_accu, _unknown_args)
