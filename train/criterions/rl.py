import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()


class A2CLoss(nn.Module):
    def __init__(self, returns_normalize=True, use_V=True, eta=0.01) -> None:
        """
        compute the objective node for policy/value networks

        Parameters
        ----------
        probs : list
            action prob at time t
        values : list
            state value at time t
        rewards : list
            rewards at time t

        Returns
        -------
        torch.tensor, torch.tensor
            Description of returned object.

        """
        super().__init__()
        self.returns_normalize = returns_normalize
        self.use_V = use_V
        self.eta = eta

    def forward(self, probs, values, rewards, entropys, device='cpu'):
        returns = compute_returns(rewards, normalize=self.returns_normalize)
        policy_grads, value_losses = [], []
        for prob_t, v_t, R_t in zip(probs, values, returns):
            if self.use_V:
                A_t = R_t - v_t.item()
                value_losses.append(
                    # smooth_l1_loss(torch.squeeze(v_t.to(self.device)), torch.squeeze(R_t.to(self.device)))
                    0.5 * mse_loss(torch.squeeze(v_t.to(device)), torch.squeeze(R_t.to(device)))
                )
            else:
                A_t = R_t
                value_losses.append(torch.FloatTensor(0).data)
            # accumulate policy gradient
            policy_grads.append(-prob_t * A_t)
        policy_gradient = torch.stack(policy_grads).mean()
        value_loss = torch.stack(value_losses).mean()
        pi_ent = torch.stack(entropys).mean()
        loss = policy_gradient + value_loss - pi_ent * self.eta
        return loss, policy_gradient, value_loss


def pick_action(action_distribution):
    """action selection by sampling from a multinomial.

    Parameters
    ----------
    action_distribution : 1d torch.tensor
        action distribution, pi(a|s)

    Returns
    -------
    torch.tensor(int), torch.tensor(float)
        sampled action, log_prob(sampled action)

    """
    m = Categorical(action_distribution)
    a_t = m.sample()
    a_t_max = torch.argmax(action_distribution)
    log_prob_a_t = m.log_prob(a_t)
    return a_t, log_prob_a_t, a_t_max


def compute_returns(rewards, gamma=0, normalize=False):
    """
    compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 1d array
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    1d torch.tensor
        the sequence of cumulative return

    """
    # compute cumulative discounted reward since t, for all t
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # normalize w.r.t to the statistics of this trajectory
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns
