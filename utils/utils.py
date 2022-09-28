import importlib
import pathlib
import torch
import matplotlib.pyplot as plt

from .dict_utils import load_dict


def import_attr(module_and_attr_name):
    """
    Import an attr (class or function) from a (python) module, e.g. 'models.RNN' (taken from the Full Stack DL course)
    Args:
        module_and_attr_name: if list of str, try to import each attr in order and return the first that can be imported
    """
    module_name, class_name = module_and_attr_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    attr_ = getattr(module, class_name)

    return attr_


def savefig(save_dir, filename, pdf=False):
    if save_dir is None:
        return None
    if isinstance(save_dir, str):
        save_dir = pathlib.Path(save_dir)

    filename = filename.replace('.', '-')
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir/filename)
    plt.savefig(save_dir/(filename + ".pdf"), transparent=True) if pdf else None
    plt.close()


def load_setup(setup_path):
    setup = load_dict(setup_path)
    return setup


def parse_setup(setup, device):
    model = load_model(setup.pop("model"), device)
    setup["model_name"] = model.__class__.__name__

    training_setups = setup.pop("training")
    envs, optimizers, schedulers, criterions = [], [], [], []
    for training_setup in training_setups:
        if "env" in training_setup:
            env = load_environment(training_setup.pop("env"))
        else:
            env = None
        if "trainer" in training_setup:
            optimizer, scheduler = load_optimizer(training_setup["trainer"].pop("optimizer"), model)
            criterion = load_criterion(training_setup["trainer"].pop("criterion"))
        else:
            optimizer, scheduler, criterion = None, None, None
        envs.append(env)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        criterions.append(criterion)
    
    return model, envs, optimizers, schedulers, criterions, training_setups, setup


def load_model(setup, device):
    subclasses_dict = {}
    if "subclasses" in setup:
        for subclass in setup["subclasses"]:
            name = subclass.pop("name")
            submodel = load_model(subclass, device)
            subclasses_dict[name] = submodel
        setup.pop("subclasses")
    model_class = setup.pop("class")
    for key, value in subclasses_dict.items():
        setup[key] = value
    model = import_attr("models.{}".format(model_class))(device=device, **setup)
    return model


def load_environment(setup):
    task_class = setup.pop("class")
    env = import_attr("tasks.{}".format(task_class))(**setup)
    return env


def load_optimizer(setup, model):
    optimizer_class = setup.pop("class")
    optimizer = import_attr("torch.optim.{}".format(optimizer_class))(model.parameters(), lr=setup["lr"], weight_decay=setup.get("weight_decay", 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=setup.get("lr_decay_factor", 1e-3), patience=setup.get("lr_decay_patience", 30), 
        threshold=setup.get("lr_decay_threshold", 1e-3), min_lr=setup.get("min_lr", 1e-8), verbose=True)
    return optimizer, scheduler


def load_criterion(setup):
    criterion_name = setup.pop("class")
    if hasattr(torch.nn, criterion_name):
        criterion = import_attr("torch.nn.{}".format(criterion_name))(**setup)
    else:
        criterion = import_attr("train.criterions.{}".format(criterion_name))(**setup)
    return criterion
