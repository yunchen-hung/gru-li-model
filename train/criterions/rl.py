from math import gamma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()


class A2CLoss(nn.Module):
    def __init__(self, returns_normalize=False, use_V=True, eta=0.001, gamma=0.9) -> None:
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
        self.gamma = gamma

    def forward(self, probs, values, rewards, entropys, device='cpu'):
        returns = compute_returns(rewards, gamma=self.gamma, normalize=self.returns_normalize)
        policy_grads, value_losses = [], []
        # print(prob_t, v_t, R_t)
        # print(torch.tensor(probs).shape)
        # print(torch.stack(values).shape)
        # print(torch.stack(returns).shape)
        probs, values, rewards, entropys = torch.stack([torch.stack(prob_t).to(device) for prob_t in probs]).transpose(1, 0).to(device), \
                                torch.stack(values).squeeze(2).transpose(1, 0).to(device), \
                                torch.stack(returns).to(device), \
                                torch.stack([torch.stack(entropys_t) for entropys_t in entropys])
        # print(probs)
        # print(values)
        # print(rewards)
        # print(entropys)
        if self.use_V:
            A = rewards - values.item()
            value_losses = 0.5 * mse_loss(torch.squeeze(values.to(device)), torch.squeeze(rewards.to(device)))
            # smooth_l1_loss(torch.squeeze(v_t.to(self.device)), torch.squeeze(R_t.to(self.device)))
        else:
            A = rewards
            value_losses = torch.tensor(0.0).to(device)
        # accumulate policy gradient
        policy_grads = -probs * A
        policy_gradient = torch.mean(policy_grads)
        value_loss = torch.mean(value_losses)
        pi_ent = torch.mean(entropys)
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
    a_ts, log_prob_a_ts, a_t_maxs = [], [], []
    for i in range(action_distribution.shape[0]):
        m = Categorical(action_distribution[i])
        a_t = m.sample()
        a_t_max = torch.argmax(action_distribution[i])
        log_prob_a_t = m.log_prob(a_t)
        a_ts.append(a_t)
        log_prob_a_ts.append(log_prob_a_t)
        a_t_maxs.append(a_t_max)
    return a_ts, log_prob_a_ts, a_t_maxs


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
    R = 0.0
    rewards = np.array(rewards)
    returns_all = []
    for i in range(rewards.shape[1]):
        reward = rewards[:, i]
        returns = []
        for r in reward[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # normalize w.r.t to the statistics of this trajectory
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + eps)
        returns_all.append(returns)
    return returns_all
