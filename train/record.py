import torch
from models.basic_module import analyze

from models.rl import pick_action, compute_returns, compute_a2c_loss
from models.utils import entropy
from .utils import count_accuracy, save_model


def record_model(agent, env, num_trials_per_condition=100, device='cpu'):
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, total_critic_loss = 0.0, 0, 0, 0, 0.0, 0.0, 0.0
    test_accuracies = []
    test_errors = []

    context_num = env.context_num
    num_iter = context_num * num_trials_per_condition

    agent.set_retrieval(True)

    for i in range(num_trials_per_condition):
        for j in range(context_num):
            actions, probs, rewards, values, entropys = [], [], [], [], []
            # inps, outs = {}, {}
            # def forward_hook(module, input, output):
            #     print(module)
            #     inps[id(module)] = input
            #     outs[id(module)] = output

            obs = torch.Tensor(env.reset(context_index=j)).to(device)
            done = False
            
            # hook = agent.register_forward_hook(forward_hook)
            agent.set_encoding(False)
            state = agent.init_state(1)
            while not done:
                with analyze(agent):
                    action_distribution, value, state = agent(obs, state)
                readout = agent.readout()
                # print(readout.keys())
                # print(readout['LSTM'].keys())
                # print(readout['ValueMemory'].keys())
                # print(readout['ValueMemory']['LCASimilarity'].keys())
                action, log_prob_action = pick_action(action_distribution)
                obs_, reward, done, info = env.step(action)
                obs = torch.Tensor(obs_).to(device)

                if info.get("encoding_on", False):
                    agent.set_encoding(True)

                probs.append(log_prob_action)
                rewards.append(reward)
                values.append(value)
                entropys.append(entropy(action_distribution))
                actions.append(action)    
                total_reward += reward
            # hook.remove()
            # print("hook:", inps.keys(), outs.keys())
            break

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
        break

    accuracy = actions_correct_num / actions_total_num
    error = actions_wrong_num / actions_total_num
    not_know_rate = 1 - accuracy - error
    mean_reward = total_reward / actions_total_num
    mean_loss = total_loss / num_iter
    mean_actor_loss = total_actor_loss / num_iter
    mean_critic_loss = total_critic_loss / num_iter
    return accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss
