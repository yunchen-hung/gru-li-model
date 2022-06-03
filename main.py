import os
import argparse
from pathlib import Path
import torch

import consts
from utils import load_setup, parse_setup, import_attr
from train import train_model, plot_accuracy_and_error, record_model


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
    setup = load_setup(exp_dir/consts.SETUP_FOLDER/setup_name)

    model, env, optimizer, scheduler, setup = parse_setup(setup, device)

    run_name = setup.get("run_name", setup_name.split(".")[0])
    print("run_name: {}".format(run_name))

    model_save_path = exp_dir/consts.SAVE_MODEL_FOLDER/setup["model_name"]/run_name
    model_save_path.mkdir(parents=True, exist_ok=True)

    # print(model)
    # for child in model.children():
    #     print(child)

    if not train and os.path.exists(model_save_path/"model.pt"):
        model.load_state_dict(torch.load(model_save_path/"model.pt"))

    print("device: ", device)
    model.to(device)

    if train or not os.path.exists(model_save_path/"model.pt"):
        test_accuracies, test_errors = train_model(model, env, optimizer, scheduler, setup, device=device, model_save_path=model_save_path, **setup["training"])
        plot_accuracy_and_error(test_accuracies, test_errors, model_save_path)

    data = record_model(model, env, device=device)

    paths = {"fig": exp_dir/consts.FIGURE_FOLDER/setup["model_name"]/run_name}

    run_exp = import_attr("{}.{}.experiment.run".format(consts.EXPERIMENT_FOLDER.replace('/', '.'), experiment))
    run_exp(data, model, env, paths)


if __name__ == "__main__":
    _experiment, _setup_name, _device, _train, _debug, _test_accu, _unknown_args = parse_args()
    main(_experiment, _setup_name, _device, _train, _debug, _test_accu, _unknown_args)
