import time
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import count_accuracy, save_model
from torch.nn.functional import mse_loss


def supervised_train_model(agent, env, optimizer, scheduler, setup, criterion, sl_criterion=None, 
    num_iter=10000, test=False, test_iter=200, save_iter=1000, stop_test_accu=1.0, 
    device='cpu', model_save_path=None, use_memory=None, min_iter=0, batch_size=1, phase="encoding",
    memory_entropy_reg=False, memory_reg_weight=0.0, ):

    num_iter, test_iter, save_iter = int(num_iter), int(test_iter), int(save_iter)
    
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
        actions, probs, rewards, actions_max, outputs, outputs2, mem_similarities, mem_sim_entropys = [], [], [], [], [], [], [], []

        # reset environment
        obs_, info = env.reset(batch_size)
        obs = torch.Tensor(obs_).to(device)
        # print(env.memory_sequence)
        done = np.zeros(batch_size, dtype=bool)
        # print(obs)

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
                # print("reset state before recall")
                state = agent.init_state(batch_size, recall=True, prev_state=state)

            # do one step of forward pass for the agent
            output, _, output2, _, state, mem_similarity = agent(obs, state)
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
            # print(action)
            # print(obs, reward, info)

            probs.append(log_prob_action)
            rewards.append(reward)
            actions.append(action)
            actions_max.append(action_max)
            outputs.append(output)
            outputs2.append(output2)
            mem_similarities.append(mem_similarity)
            mem_sim_entropys.append(entropy(mem_similarity, device))
        # print()
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
            if outputs2[0] is not None:
                outputs2 = torch.stack(outputs2)

        gt, mask = env.get_ground_truth(phase=phase)
        gt = torch.tensor(gt).to(device)
        mask = torch.tensor(mask).to(device)
        gt = gt[mask == 1].reshape(1, -1)
        # print(gt)

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
        actions_tensor = torch.stack(actions_max).reshape(1, -1)
        # print(actions_tensor, gt, mask)
        actions_tensor = actions_tensor[mask == 1].reshape(1, -1)
        # print(actions_tensor.shape)
        # print(actions_tensor, gt)
        correct_actions = torch.sum(actions_tensor == gt)
        wrong_actions = torch.sum(actions_tensor != gt)
        not_know_actions = 0
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        # print(outputs.shape, gt.shape)
        # gt = gt.reshape(1, -1)
        loss = criterion(outputs[mask.reshape(-1, 1) == 1], gt.permute(1, 0), memory_num=memory_num)
        if sl_criterion is not None and outputs2[0] is not None:
            loss += sl_criterion(outputs2[mask.reshape(-1, 1) == 1], gt.permute(1, 0), memory_num=memory_num)
        if memory_entropy_reg:
            # add (negative) entropy regularization for memory similarity
            # to encourage the memory similarity to be closer to one-hot
            # print([len(mem_sim_ent) for mem_sim_ent in mem_sim_entropys])
            mem_ent_reg_loss = memory_reg_weight * torch.mean(torch.stack([torch.stack(mem_sim_ent) for mem_sim_ent in mem_sim_entropys[memory_num:]]))
            loss += mem_ent_reg_loss
        else:
            mem_ent_reg_loss = None

        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % test_iter == 0:

            env.render()

            # print(gt.shape, np.array(actions).shape)
            print("gt, encoding action:", gt.detach().cpu().numpy(), 
                    torch.argmax(outputs[mask.reshape(-1, 1) == 1], dim=1).detach().cpu().numpy().reshape(-1))
            if sl_criterion is not None and outputs2[0] is not None:
                print("gt2, encoding action2:", gt.detach().cpu().numpy(), 
                    torch.argmax(outputs2[mask.reshape(-1, 1) == 1], dim=1).detach().cpu().numpy().reshape(-1))
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

            print()

            if i != 0:
                scheduler.step(total_loss)  # TODO: change a criterion here?

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
