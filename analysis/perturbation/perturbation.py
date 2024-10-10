import numpy as np
import torch
import matplotlib.pyplot as plt

from train.criterions.rl import pick_action
from utils import savefig



class Perturbation:
    def __init__(self, noise_start=0.0, noise_end=1.0, noise_step=0.05):
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.noise_step = noise_step
        self.results = None

    def fit(self, model, env, coefs, noise_start=None, noise_end=None, noise_step=None, trial_num=1000):
        self.noise_start = noise_start if noise_start is not None else self.noise_start
        self.noise_end = noise_end if noise_end is not None else self.noise_end
        self.noise_step = noise_step if noise_step is not None else self.noise_step
        accuracies = []
        for i in np.arange(self.noise_start, self.noise_end, self.noise_step):
            accuracy = record_with_noise(model, env, coefs, noise_proportion=i, trial_num=trial_num)
            print(f"noise proportion: {i}, accuracy: {accuracy}")
            accuracies.append(accuracy)
        self.results = accuracies
        return accuracies

    def visualize(self, save_path, save_name="perturbation", title=None, xlabel="noise proportion", 
                  ylabel="performance", figsize=None, format="png"):
        figsize = figsize if figsize is not None else (4.0, 3.3)
        plt.figure(figsize=figsize, dpi=180)

        plt.plot(np.arange(self.noise_start, self.noise_end, self.noise_step), self.results)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path is not None:
            savefig(save_path, save_name, format=format)



def record_with_noise(agent, env, coefs, noise_proportion=0.05, trial_num=1000, used_output=0, 
                      reset_memory=True, device='cpu'):
    accuracy = 0.0
    correct_actions, wrong_actions, not_know_actions = 0, 0, 0
    batch_size = 1 # env.num_envs

    for i in range(coefs.shape[0]):
        if not np.std(coefs[i]) == None and not np.std(coefs[i]) == 0:
            coefs[i] = coefs[i] / np.std(coefs[i])
            # print(np.std(coefs[i]))
    coefs = torch.Tensor(coefs).to(device)

    for i in range(trial_num):
        obs_, info = env.reset()
        agent.reset_memory(flush=reset_memory)
        obs = torch.Tensor(obs_).reshape(1, -1).to(device)
        done = False
        state = agent.init_state(batch_size)

        while not done:
            # set up the phase of the agent
            if info["phase"] == "encoding":
                agent.set_encoding(True)
                agent.set_retrieval(False)
            elif info["phase"] == "recall":
                agent.set_encoding(False)
                agent.set_retrieval(True)
            # reset state between phases
            if "reset_state" in info and info["reset_state"]:
                state = agent.init_state(batch_size, recall=True, prev_state=state)

            # generate noise and add it to hidden state
            if not torch.mean(state) == 0:
                state_std = torch.std(state, dim=1)
                for i in range(state.shape[0]):
                    state = state * (1-noise_proportion) + torch.randn_like(state) * coefs[i] * state_std[i] / state.shape[0] * noise_proportion
            
            output, value, state, _ = agent(obs, state)
            action_distribution = output[used_output]
            action, log_prob_action, action_max = pick_action(action_distribution)
            obs_, reward, _, _, info = env.step(action)
            done = info["done"]
            obs = torch.Tensor(obs_).reshape(1, -1).to(device)

            correct_actions += np.sum(info["correct"])
            wrong_actions += np.sum(info["wrong"])
            not_know_actions += np.sum(info["not_know"])

    accuracy = correct_actions / (correct_actions + wrong_actions + not_know_actions)
    
    return accuracy
    
