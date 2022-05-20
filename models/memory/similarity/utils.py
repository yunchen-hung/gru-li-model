import torch


def make_weights(diag_val, offdiag_val, weight_size, device):
    """Get a connection weight matrix with "diag-offdial structure"

    e.g.
        | x, y, y |
        | y, x, y |
        | y, y, x |

    where x = diag_val, and y = offdiag_val

    Parameters
    ----------
    diag_val : float
        the value of the diag entries
    offdiag_val : float
        the value of the off-diag entries
    weight_size : int
        the number of LCA nodes

    Returns
    -------
    2d array
        the weight matrix with "diag-offdial structure"

    """
    diag_mask = torch.eye(weight_size)
    offdiag_mask = torch.ones((weight_size, weight_size)) - torch.eye(weight_size)
    weight_matrix = diag_mask * diag_val + offdiag_mask * offdiag_val
    return weight_matrix.float().to(device)
