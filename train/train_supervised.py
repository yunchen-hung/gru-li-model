import time
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import count_accuracy, save_model
from torch.nn.functional import mse_loss


def supervised_train_model(agent, env, optimizer, scheduler, setup, criterion, num_iter=10000, test=False, test_iter=200, save_iter=1000, stop_test_accu=1.0, 
    device='cpu', model_save_path=None, use_memory=None, min_iter=0, batch_size=1):
    actions_correct_num, actions_wrong_num, actions_total_num, total_loss = 0, 0, 0, 0.0
    test_accuracies = []
    test_errors = []

    if use_memory:
        agent.use_memory = use_memory
    if agent.use_memory:
        print("Agent use memory")
    else:
        print("Agent not use memory")

    print("start supervised training")
    print("batch size:", batch_size)
    min_test_loss = torch.inf

    for i in range(num_iter):
        state = agent.init_state(batch_size)
        agent.reset_memory()
        agent.set_encoding(False)
        agent.set_retrieval(False)

        # create variables to store data related to outputs and results
        actions, probs, rewards, actions_max, outputs = [], [], [], [], []

        # reset environment
        obs_, info = env.reset(batch_size)
        obs = torch.Tensor(obs_).to(device)
        # print(env.memory_sequence)
        done = np.zeros(batch_size, dtype=bool)

        memory_num = 0

        while not done.all():
            # set up the phase of the agent
            if info["phase"] == "encoding":
                memory_num += 1
                agent.set_encoding(True)
                agent.set_retrieval(False)
            elif info["phase"] == "recall":
                agent.set_encoding(False)
                agent.set_retrieval(True)
            # reset state between phases
            if info.get("reset_state", False):
                state = agent.init_state(batch_size, recall=True, prev_state=state)

            # do one step of forward pass for the agent
            output, _, state, _ = agent(obs, state)
            if isinstance(output, tuple):
                # when generating two decisions, only record the first one as action
                action_distribution = output[0]
            else:
                action_distribution = output
            action, log_prob_action, action_max = pick_action(action_distribution)
            # info_ = info
            obs_, reward, done, info = env.step(action)
            # print(obs, action, reward, info_)
            obs = torch.Tensor(obs_).to(device)

            probs.append(log_prob_action)
            rewards.append(reward)
            actions.append(action)
            actions_max.append(action_max)
            outputs.append(output)
        if isinstance(outputs[0], tuple):
            outputs_ts = [[] for _ in range(len(outputs[0]))]
            for output in outputs:
                for j, o in enumerate(output):
                    outputs_ts[j].append(o)
            for j in range(len(outputs_ts)):
                outputs_ts[j] = torch.stack(outputs_ts[j])
            outputs = tuple(outputs_ts)
        else:
            outputs = torch.stack(outputs)

        gt = torch.tensor(env.get_ground_truth(phase="encoding")).to(device)

        # if random_action:
        #     correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(np.array([action[0].item() for action in actions]))
        # else:
        #     correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(np.array([action[0].item() for action in actions_max]))
        # # rewards = env.compute_rewards(actions)
        # # print(torch.stack(actions[env.memory_num:]).detach().cpu().numpy(), env.memory_sequence, correct_actions, wrong_actions, not_know_actions)
        # actions_total_num += correct_actions + wrong_actions + not_know_actions
        # actions_correct_num += correct_actions
        # actions_wrong_num += wrong_actions

        # actions_tensor = torch.stack([torch.stack(action) for action in actions]).reshape(-1)[:memory_num]
        # print(torch.stack(actions).shape, gt.shape)
        actions_tensor = torch.stack(actions)[:memory_num]
        correct_actions = torch.sum(actions_tensor.T == gt)
        wrong_actions = torch.sum(actions_tensor.T != gt)
        not_know_actions = 0
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        # print(outputs.shape, gt.shape)
        loss = criterion(outputs[:memory_num], gt, memory_num=memory_num)

        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % test_iter == 0:

            env.render()

            # print(gt.shape, np.array(actions).shape)
            print("gt, action, encoding action:", gt[0].detach().cpu().numpy(), np.array(actions)[:memory_num,0], 
                    torch.argmax(outputs[:memory_num, 0], dim=1).detach().cpu().numpy().reshape(-1))
            accuracy = actions_correct_num / actions_total_num
            error = actions_wrong_num / actions_total_num
            not_know_rate = 1 - accuracy - error
            mean_loss = total_loss / test_iter

            print('Supervised, Iteration: {},  train accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, '
            'total loss: {:.2f}'.format(i, accuracy, error, not_know_rate, mean_loss))

            if test:
                test_accuracy, test_error, test_not_know_rate, test_mean_reward, test_mean_loss, test_mean_actor_loss, test_mean_critic_loss \
                    = count_accuracy(agent, env, num_trials_per_condition=10, device=device)

                print('\ttest accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.2f}, actor loss: {:.2f}, critic loss: {:.2f}'\
                    .format(test_accuracy, test_error, test_not_know_rate, test_mean_reward, test_mean_loss, test_mean_actor_loss, test_mean_critic_loss))
            else:
                test_error = error
                test_accuracy = accuracy
                test_mean_loss = mean_loss

            if i != 0:
                scheduler.step(test_error - test_accuracy)  # TODO: change a criterion here?

            if test_error - test_accuracy <= min_test_loss:
                min_test_loss = test_error - test_accuracy
                save_model(agent, model_save_path, filename="model.pt")
            
            if test_accuracy >= stop_test_accu and i > min_iter:
                break

            test_accuracies.append(test_accuracy)
            test_errors.append(test_error)

            actions_correct_num, actions_wrong_num, actions_total_num, total_loss = 0, 0, 0, 0.0
        
        if i % save_iter == 0:
            save_model(agent, model_save_path, filename="sup_{}.pt".format(i))
    
    return test_accuracies, test_errors
