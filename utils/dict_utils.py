import collections.abc
import json


def update(d, u):
    """
    Recursively update dict (https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth)
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_dict(dict_path, d=None):
    """
    Recursively load and parse dict
    Args:
        dict_path: path to a json file
        d (optional): dict or subdict loaded from dict_path
    """
    if d is None:
        # load json file
        with open(dict_path, 'r') as f:
            d = json.load(f)

    if not isinstance(d, dict):
        raise ValueError("A dict instance is expected")

    # parse base dict
    if 'base_dict' in d:
        base_dict = load_dict(dict_path.parents[0] / d['base_dict'])
        del d['base_dict']
        # overwrite base dict
        # base_dict.update(d)
        update(base_dict, d)
        d = base_dict

    # parse subdicts
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = load_dict(dict_path, d=v)
    return d


def get_dict_item(dict, path):
    """
    Get an item from a dict using a path (list of keys)
    """
    if isinstance(path, str):
        path = path.split(".")
    for key in path:
        dict = dict[key]
    return dict


def set_dict_item(dict, path, value):
    """
    Set an item in a dict using a path (list of keys)
    """
    if isinstance(path, str):
        path = path.split(".")
    for key in path[:-1]:
        dict = dict[key]
    dict[path[-1]] = value
