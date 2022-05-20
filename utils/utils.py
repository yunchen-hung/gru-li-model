import importlib
import torch

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


def load_setup(setup_path):
    setup = load_dict(setup_path)
    return setup


def parse_setup(setup):
    model = load_model(setup.pop("model"))
    env = load_environment(setup.pop("env"))
    optimizer, scheduler = load_optimizer(setup["training"].pop("optimizer"), model)
    setup["model_name"] = model.__class__.__name__
    return model, env, optimizer, scheduler, setup


def load_model(setup):
    subclasses_dict = {}
    if "subclasses" in setup:
        for subclass in setup["subclasses"]:
            name = subclass.pop("name")
            submodel = load_model(subclass)
            subclasses_dict[name] = submodel
        setup.pop("subclasses")
    model_class = setup.pop("class")
    for key, value in subclasses_dict.items():
        setup[key] = value
    model = import_attr("models.{}".format(model_class))(**setup)
    return model


def load_environment(setup):
    task_class = setup.pop("class")
    env = import_attr("tasks.{}".format(task_class))(**setup)
    return env


def load_optimizer(setup, model):
    optimizer_class = setup.pop("class")
    optimizer = import_attr("torch.optim.{}".format(optimizer_class))(model.parameters(), lr=setup["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=setup.get("lr_decay_factor", 1e-3), patience=setup.get("lr_decay_patience", 30), 
        threshold=setup.get("lr_decay_threshold", 1e-3), min_lr=setup.get("min_lr", 1e-8), verbose=True)
    return optimizer, scheduler
