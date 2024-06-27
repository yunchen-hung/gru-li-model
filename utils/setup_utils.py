import torch
import matplotlib.pyplot as plt
import itertools
import copy
import gymnasium as gym

from .dict_utils import load_dict, get_dict_item, set_dict_item
from .utils import import_attr


def load_setup(setup_path):
    setup = load_dict(setup_path)
    return setup


def parse_setup(general_setup, device):
    setups = parse_vary_params(general_setup)

    model_instances = {}

    for run_name, setup in setups.items():
        model = load_model(setup.pop("model"), device)
        setup["model_name"] = model.__class__.__name__

        if "model_for_record" in setup:
            model_for_record = load_model(setup.pop("model_for_record"), device)
        else:
            model_for_record = None

        training_setups = setup.pop("training")
        envs, optimizers, schedulers, criterions, sl_criterions = [], [], [], [], []
        for training_setup in training_setups:
            if "env" in training_setup:
                env = load_environment(training_setup.pop("env"))
            else:
                env = None
            if "trainer" in training_setup:
                optimizer, scheduler = load_optimizer(training_setup["trainer"].pop("optimizer"), model)
                criterion = load_criterion(training_setup["trainer"].pop("criterion"))
                if "sl_criterion" in training_setup["trainer"]:
                    sl_criterion = load_criterion(training_setup["trainer"].pop("sl_criterion"))
                else:
                    sl_criterion = None
            else:
                optimizer, scheduler, criterion, sl_criterion = None, None, None, None
            envs.append(env)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            criterions.append(criterion)
            sl_criterions.append(sl_criterion)
        model_instances[run_name] = model, model_for_record, envs, optimizers, schedulers, criterions, sl_criterions, training_setups, setup
    
    return model_instances


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
    envs = []
    for env_setup in setup:
        if "vector_env" in env_setup:
            mode = env_setup["vector_env"]["mode"]
            batch_size = env_setup["vector_env"]["batch_size"]
            env_setup.pop("vector_env")
            if mode == "sync":
                env = gym.vector.SyncVectorEnv([
                    lambda: load_single_environment(env_setup)
                    for _ in range(batch_size)
                ])
            elif mode == "async":
                env = gym.vector.AsyncVectorEnv([
                    lambda: load_single_environment(env_setup)
                    for _ in range(batch_size)
                ])
            else:
                raise AttributeError("vector env mode must be 'async' or 'sync'")
        else:
            env = load_single_environment(env_setup)
        envs.append(env)
    return envs


def load_single_environment(setup):
    task_class = setup.pop("class")
    if "wrapper" in setup:
        wrapper_setups = setup.pop("wrapper")
        env = import_attr("tasks.{}".format(task_class))(**setup)
        for wrapper_setup in wrapper_setups:
            wrapper_class = wrapper_setup.pop("class")
            env = import_attr("tasks.wrappers.{}".format(wrapper_class))(env, **wrapper_setup)
    else:
        env = import_attr("tasks.{}".format(task_class))(**setup)
    return env
        

def load_optimizer(setup, model):
    optimizer_class = setup.pop("class")
    optimizer = import_attr("torch.optim.{}".format(optimizer_class))(model.parameters(), lr=setup["lr"], weight_decay=setup.get("weight_decay", 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=setup.get("lr_decay_factor", 1e-3), patience=setup.get("lr_decay_patience", 30), 
        threshold=setup.get("lr_decay_threshold", 1e-3), min_lr=setup.get("min_lr", 1e-8))
    return optimizer, scheduler


def load_criterion(setup):
    # subclasses_dict = {}
    # if "subclasses" in setup:
    #     for subclass in setup["subclasses"]:
    #         name = subclass.pop("name")
    #         submodel = load_criterion(subclass)
    #         subclasses_dict[name] = submodel
    #     setup.pop("subclasses")
    # for key, value in subclasses_dict.items():
    #     setup[key] = value
    if "criteria" in setup:
        criteria = []
        for criterion in setup["criteria"]:
            criterion = load_criterion(criterion)
            criteria.append(criterion)
        setup.pop("criteria")
        setup["criteria"] = criteria
    criterion_name = setup.pop("class")
    if hasattr(torch.nn, criterion_name):
        criterion = import_attr("torch.nn.{}".format(criterion_name))(**setup)
    else:
        criterion = import_attr("train.criterions.{}".format(criterion_name))(**setup)
    return criterion


def parse_vary_params(setup):
    setups = {}
    general_run_name = setup.pop("run_name", "")
    if "vary_params" in setup.keys() and "combinatorial" in setup["vary_params"].keys():
        print("vary_params")
        vary_params = setup.pop("vary_params")
        combinatorial = vary_params["combinatorial"]
        combinatorial_name = vary_params.get("combinatorial_name", combinatorial)
        assert isinstance(combinatorial, list) and len(combinatorial) > 0, "Combinatorial must be a non-empty list"
        assert isinstance(combinatorial_name, list) and len(combinatorial_name) > 0, "Combinatorial_name must be a non-empty list"
        assert len(combinatorial) == len(combinatorial_name), "Combinatorial and combinatorial_name must have the same length"
        # TODO: add sequential
        params = []
        for param_path in combinatorial:
            param = get_dict_item(setup, param_path)
            assert isinstance(param, list) and len(param) != 0, "Combinatorial parameters must be a non-empty list"
            params.append(param)
        params_len = [range(len(param)) for param in params]
        for i in itertools.product(*params_len):
            s = copy.deepcopy(setup)
            index_list = list(i)
            run_name = general_run_name
            for j, k in enumerate(index_list):
                set_dict_item(s, combinatorial[j], params[j][k])
                run_name = run_name + '_' + combinatorial_name[j] + str(params[j][k])
            setups[run_name] = s
    else:
        setups[general_run_name] = setup
    return setups
