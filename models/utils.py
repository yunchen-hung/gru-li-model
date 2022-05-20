import torch

from utils import import_attr


def load_act_fn(act_fn: str):
    if act_fn == "ReTanh":
        return lambda x: torch.max(torch.tanh(x), torch.zeros_like(x))
    else:
        return import_attr("torch.nn.{}".format(act_fn))()


def entropy(probs):
    """calculate entropy.
    I'm using log base 2!

    Parameters
    ----------
    probs : a torch vector
        a prob distribution

    Returns
    -------
    torch scalar
        the entropy of the distribution

    """
    return - torch.stack([pi * torch.log2(pi) for pi in probs]).sum()
