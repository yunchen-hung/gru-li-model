import time
from collections import defaultdict
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import count_accuracy, save_model
from torch.nn.functional import mse_loss


def supervised_train_model(agent, envs, optimizer, scheduler, setup, criterion, sl_criterion=None, 
    num_iter=10000, test=False, test_iter=200, save_iter=1000, stop_test_accu=1.0, 
    device='cpu', model_save_path=None, use_memory=None, min_iter=0, batch_size=1, phase="encoding",
    memory_entropy_reg=False, memory_reg_weight=0.0, reset_memory=True, 
    used_output_index=[0], env_sample_prob=[1.0]):

    # each used_output_index corresponds to an environment
    # specifying the output that is used for computing the action for stepping the environment
    assert len(envs) == len(used_output_index) == len(env_sample_prob)

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
        agent.reset_memory(flush=reset_memory)
        agent.set_encoding(False)
        agent.set_retrieval(False)

        # create variables to store data related to outputs and results
        outputs = defaultdict(list)
        actions, probs, actions_max = defaultdict(list), defaultdict(list), defaultdict(list)
        rewards, mem_similarities, mem_sim_entropys = [], [], []

        # randomly sample an environment from the list of environments
        env_id = np.random.choice(len(envs), p=env_sample_prob)
        env = envs[env_id]

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
            output, _, state, mem_similarity = agent(obs, state)
            env_updated = False
            for j, o in enumerate(output):
                action_distribution = o
                action, log_prob_action, action_max = pick_action(action_distribution)
                if j == used_output_index[env_id]:
                    obs_, reward, done, info = env.step(action)
                    obs = torch.Tensor(obs_).to(device)
                    rewards.append(reward)
                    env_updated = True
                probs[j].append(log_prob_action)
                actions[j].append(action)
                actions_max[j].append(action_max)
                outputs[j].append(o)
            assert env_updated

            mem_similarities.append(mem_similarity)
            mem_sim_entropys.append(entropy(mem_similarity, device))

        gt, mask = env.get_ground_truth(phase=phase)
        gt = torch.tensor(gt).to(device)
        mask = torch.tensor(mask).to(device)
        gt = gt[mask == 1].reshape(1, -1)

        actions_tensor = torch.stack(actions_max[used_output_index[env_id]]).reshape(1, -1)
        actions_tensor = actions_tensor[mask == 1].reshape(1, -1)
        correct_actions = torch.sum(actions_tensor == gt)
        wrong_actions = torch.sum(actions_tensor != gt)
        not_know_actions = 0
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        for j in range(len(outputs)):
            outputs[j] = torch.stack(outputs[j])[mask.reshape(-1, 1) == 1]

        loss = criterion(outputs, gt.permute(1, 0))
        if memory_entropy_reg:
            # add (negative) entropy regularization for memory similarity
            # to encourage the memory similarity to be closer to one-hot
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

            print("current env:", env_id)
            env.render()

            # print(gt.shape, np.array(actions).shape)
            for j in range(len(outputs)):
                print("gt{}, action{}:".format(j+1, j+1), gt.detach().cpu().numpy(), 
                    torch.argmax(outputs[j], dim=1).detach().cpu().numpy().reshape(-1))
                
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
