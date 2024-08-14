from math import gamma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()


class A2CLoss(nn.Module):
    def __init__(self, returns_normalize=False, use_V=True, eta=0.0, gamma=0.0, phase='recall', value_loss_func='l1') -> None:
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
        self.phase = phase
        self.value_loss_func = value_loss_func

    def forward(self, probs, values, rewards, entropys, print_info=False, device='cpu'):
        """
        probs: list of torch.tensor, overall size is (timesteps, batch_size)
        values: list of torch.tensor, overall size is (timesteps, batch_size, 1)
        rewards, entropys: list of torch.tensor, overall size is (timesteps, batch_size)
        """
        # returns: batch_size x timesteps
        returns = compute_returns(rewards, gamma=self.gamma, normalize=self.returns_normalize)
        policy_grads, value_losses = [], []
        # probs: batch_size x timesteps
        # values, returns: batch_size x timesteps
        # entropys: timesteps x batch_size
        probs = torch.stack(probs).to(device).transpose(1, 0)
        values = torch.stack(values).squeeze(2).to(device).transpose(1, 0)
        returns = returns.to(device)
        # entropys = torch.stack(entropys).to(device)
        entropys = torch.stack([torch.stack(entropys_t) for entropys_t in entropys])
        
        batch_size = probs.shape[0]

        # probs, values, rewards, entropys = torch.stack([torch.stack(prob_t).to(device) for prob_t in probs]).transpose(1, 0).to(device), \
        #                         torch.stack(values).squeeze(2).transpose(1, 0).to(device), \
        #                         returns.to(device), \
        #                         torch.stack(entropys)
                                #torch.stack([torch.stack(entropys_t) for entropys_t in entropys])

        if print_info:
            print("loss info:", probs[0], values[0], returns[0])
        if self.use_V:
            # A2C loss
            A = returns - values.data
            if self.value_loss_func == 'mse':
                value_losses = 0.5 * mse_loss(torch.squeeze(values.to(device).float()), torch.squeeze(returns.to(device).float()))
            elif self.value_loss_func == 'l1':
                value_losses = smooth_l1_loss(torch.squeeze(values.to(device).float()), torch.squeeze(returns.to(device).float()))
            # smooth_l1_loss(torch.squeeze(v_t.to(self.device)), torch.squeeze(R_t.to(self.device)))
        else:
            # policy gradient loss
            A = returns
            value_losses = torch.tensor(0.0).to(device)
        # accumulate policy gradient
        policy_grads = -probs * A
        policy_gradient = torch.sum(policy_grads) / batch_size
        value_loss = torch.sum(value_losses) / batch_size
        pi_ent = torch.sum(entropys) / batch_size
        # policy_gradient = torch.mean(policy_grads)
        # value_loss = torch.mean(value_losses)
        # pi_ent = torch.mean(entropys)
        loss = policy_gradient + value_loss - pi_ent * self.eta
        return loss, policy_gradient, value_loss, pi_ent * self.eta


def pick_action(action_distribution):
    """action selection by sampling from a multinomial.

    Parameters
    ----------
    action_distribution : 2d torch.tensor, batch_size x action_dim
        action distribution, pi(a|s)

    Returns
    -------
    sampled action: 1d torch.tensor, batch_size    
    log_prob(sampled action): 2d torch.tensor, batch_size x action_dim

    """
    # a_ts, log_prob_a_ts, a_t_maxs = [], [], []
    # for i in range(action_distribution.shape[0]):
    # m = Categorical(action_distribution[i])
    m = Categorical(action_distribution)
    a_t = m.sample()
    # a_t_max = torch.argmax(action_distribution[i])
    a_t_max = torch.argmax(action_distribution, dim=1)
    log_prob_a_t = m.log_prob(a_t)
    # a_ts.append(a_t)
    # log_prob_a_ts.append(log_prob_a_t)
    # # a_t_maxs.append(a_t_max)
    # return a_ts, log_prob_a_ts, a_t_maxs
    return a_t, log_prob_a_t, a_t_max


def compute_returns(rewards, gamma=0, normalize=False):
    """
    compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 2d array, timestep x batch_size
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    2d torch.tensor, batch_size x timestep
        the sequence of cumulative return

    """
    # compute cumulative discounted reward since t, for all t
    rewards = np.array(rewards)
    returns_all = []

    R = np.zeros(rewards.shape[1])
    returns = np.zeros(rewards.shape)
    for t in range(rewards.shape[0]):
        R = rewards[-t-1] + gamma * R
        returns[-t-1] = R
    if normalize:
        returns = (returns - np.mean(returns, axis=0)) / (np.mean(returns, axis=0) + eps)
    returns = returns.T
    return torch.tensor(returns)

    # for i in range(rewards.shape[1]):
    #     reward = rewards[:, i]
    #     returns = []
    #     R = 0.0
    #     for r in reward[::-1]:
    #         R = r + gamma * R
    #         returns.insert(0, R)
    #     returns = torch.tensor(returns)
    #     # normalize w.r.t to the statistics of this trajectory
    #     if normalize:
    #         returns = (returns - returns.mean()) / (returns.std() + eps)
    #     returns_all.append(returns)
    # return returns_all
