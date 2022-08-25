import torch
import torch.nn.functional as F

from utils import import_attr


def load_act_fn(act_fn: str):
    if act_fn == "ReTanh":
        return lambda x: torch.max(torch.tanh(x), torch.zeros_like(x))
    else:
        return import_attr("torch.nn.{}".format(act_fn))()


def entropy(probs, device):
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
    return - torch.stack([pi * torch.log2(pi) if pi != 0 else torch.tensor(0, device=device) for pi in probs]).sum()

def softmax(z, beta=1.0):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    # softmax the input to a valid PMF
    pi_a = F.softmax(torch.squeeze(z / beta), dim=0)
    # make sure the output is valid
    if torch.any(torch.isnan(pi_a)):
        print(z)
        raise ValueError(f'Softmax produced nan: {z} -> {pi_a}')
    return pi_a
