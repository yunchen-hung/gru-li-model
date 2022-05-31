import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.rl import pick_action, compute_returns, compute_a2c_loss
from models.utils import entropy


def count_accuracy(agent, env, num_trials_per_condition=10, device="cpu"):
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, total_critic_loss = 0.0, 0, 0, 0, 0.0, 0.0, 0.0
    context_num = env.context_num
    num_iter = context_num * num_trials_per_condition
    for i in range(num_trials_per_condition):
        for j in range(context_num):
            actions, probs, rewards, values, entropys = [], [], [], [], []

            obs = torch.Tensor(env.reset(context_index=j)).to(device)
            done = False
            
            state = agent.init_state(1)  # TODO: possibly add batch size here
            while not done:
                action_distribution, value, state = agent.forward(obs, state)
                action, log_prob_action = pick_action(action_distribution)
                obs_, reward, done, info = env.step(action)
                obs = torch.Tensor(obs_).to(device)

                probs.append(log_prob_action)
                rewards.append(reward)
                values.append(value)
                entropys.append(entropy(action_distribution))
                actions.append(action)    
                total_reward += reward
            
            # if i < 10:
            #     print(actions, env.max_reward_arms[env.current_context_num])

            actions_total_num += len(actions)
            correct_actions, wrong_actions = env.compute_accuracy(actions)
            actions_correct_num += correct_actions
            actions_wrong_num += wrong_actions

            returns = compute_returns(rewards, normalize=True)  # TODO: make normalize a parameter
            loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
            pi_ent = torch.stack(entropys).sum()
            loss = loss_actor + loss_critic - pi_ent * 0.1  # 0.1: eta, make it a parameter

            total_loss += loss.item()
            total_actor_loss += loss_actor.item()
            total_critic_loss += loss_critic.item()

    accuracy = actions_correct_num / actions_total_num
    error = actions_wrong_num / actions_total_num
    not_know_rate = 1 - accuracy - error
    mean_reward = total_reward / actions_total_num
    mean_loss = total_loss / num_iter
    mean_actor_loss = total_actor_loss / num_iter
    mean_critic_loss = total_critic_loss / num_iter
    return accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss


def save_model(model, save_path, filename="model.pt"):
    if save_path is None:
        print("Warning: save model: No save path specified")
        return None

    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path/filename)


def plot_accuracy_and_error(accuracies, errors, save_path, filename="accuracy_and_error.png"):
    plt.figure(figsize=(12, 9))
    accuracies = np.array(accuracies)
    errors = np.array(errors)
    plt.plot(accuracies, label="accuracy")
    plt.plot(1 - errors, label="accuracy + don't know")
    plt.legend()
    plt.savefig(save_path/filename)
