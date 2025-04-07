import numpy as np
import torch

from models.base_module import analyze
from .criterions.rl import pick_action


def record(agent, env, used_output=0, context_num=20, reset_memory=False, device='cpu'):
    data = {'actions': [], 'probs': [], 'rewards': [], 'values': [], 'readouts': [],
    'trial_data': [], 'accuracy': 0.0}
    correct_actions, wrong_actions, not_know_actions = 0, 0, 0
    batch_size = 1 # env.num_envs
    for i in range(context_num):
        obs_, info = env.reset()
        agent.reset_memory(flush=reset_memory)
        obs = torch.Tensor(obs_).reshape(1, -1).to(device)
        done = False
        state = agent.init_state(batch_size)
        with analyze(agent):
            actions_trial, probs_trial, rewards_trial, values_trial = [], [], [], []
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
                
                output, value, state, _ = agent(obs, state)
                # action_distribution = output[used_output]
                action_distribution = output
                action, log_prob_action, action_max = pick_action(action_distribution)
                # print(action)
                # print(action.cpu().detach().numpy().transpose(1, 0).shape)
                obs_, reward, _, _, info = env.step(action.cpu().detach().numpy().squeeze(axis=1))
                done = info["done"]
                obs = torch.Tensor(obs_).reshape(1, -1).to(device)

                correct_actions += np.sum(info["correct"])
                wrong_actions += np.sum(info["wrong"])
                not_know_actions += np.sum(info["not_know"])

                actions_trial.append(action.detach().cpu())
                probs_trial.append(log_prob_action)
                rewards_trial.append(reward)
                # values_trial.append(value[used_output])
            readout = agent.readout()
            trial_data = env.unwrapped.get_trial_data()
            # trial_data = []
            # for single_env in env:
            #     t = single_env.get_trial_data()
            #     trial_data.append(t)

            actions_trial = torch.stack(actions_trial)
            probs_trial = torch.stack(probs_trial)
            # print(rewards_trial)
            rewards_trial = torch.Tensor(rewards_trial)
            # values_trial = torch.stack(values_trial).squeeze(-1)

            # print(actions_trial.shape, probs_trial.shape, rewards_trial.shape, values_trial.shape)
            # correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions_trial)
            # actions_total_num += correct_actions + wrong_actions + not_know_actions
            # actions_correct_num += correct_actions
            # actions_wrong_num += wrong_actions

        data['actions'].append(actions_trial)
        data['probs'].append(probs_trial)
        data['rewards'].append(rewards_trial)
        # data['values'].append(values_trial)
        data['readouts'].append(readout)
        data['trial_data'].append(trial_data)

    # print("test accuracy: {}".format(actions_correct_num/actions_total_num))
    data['accuracy'] = correct_actions / (correct_actions + wrong_actions + not_know_actions)
    data['correct_actions'] = correct_actions
    data['wrong_actions'] = wrong_actions
    data['not_know_actions'] = not_know_actions

    print("finished recording")
    
    return data
