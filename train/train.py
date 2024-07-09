import time
from collections import defaultdict
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import count_accuracy, save_model


def train(agent, envs, optimizer, scheduler, criterion, sl_criterion,
    model_save_path=None, device='cpu', use_memory=None,
    num_iter=10000, test_iter=200, save_iter=1000, min_iter=0, stop_test_accu=1.0, 
    reset_memory=True, used_output_index=[0], env_sample_prob=[1.0],
    grad_clip=True, grad_max_norm=1.0):
    """
    Train the model with RL
    """
    # each used_output_index corresponds to an environment
    # specifying the output that is used for computing the action for stepping the environment
    assert len(envs) == len(used_output_index) == len(env_sample_prob)

    # set up some parameters and variables
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, \
        total_critic_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
    test_accuracies, test_errors = [], []

    batch_size = envs[0].num_envs
    if use_memory:
        agent.use_memory = use_memory
    min_test_loss = torch.inf

    print("start training")
    print("batch size:", batch_size)

    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    
    num_iter = int(num_iter)
    loss = 0.0
    for i in range(num_iter):
        # record time for the first iteration to estimate total time needed
        if i == 0:
            start_time = time.time()

        # randomly sample an environment from the list of environments
        env_id = np.random.choice(len(envs), p=env_sample_prob)
        env = envs[env_id]
        batch_size = env.num_envs

        # before each trial, for the agent:
        # 1. reset initial state
        # 2. reset memory module
        state = agent.init_state(batch_size)
        agent.reset_memory(flush=reset_memory)

        # create variables to store data related to outputs and results
        outputs = defaultdict(list)
        actions, probs, actions_max = defaultdict(list), defaultdict(list), defaultdict(list)
        values, entropys = defaultdict(list), defaultdict(list)
        rewards, mem_similarities, mem_sim_entropys = [], [], []
        gts, gt_masks = [], []

        # reset environment
        obs_, info = env.reset()
        obs = torch.Tensor(obs_).to(device)
        # print(env.memory_sequence)
        done = np.zeros(batch_size, dtype=bool)
        memory_num = 0
        while not done.all():
            # set up the phase of the agent
            if info["phase"][0] == "encoding":
                memory_num += 1
                agent.set_encoding(True)
                agent.set_retrieval(False)
            elif info["phase"][0] == "recall":
                agent.set_encoding(False)
                agent.set_retrieval(True)
            # reset state between phases
            if "reset_state" in info and info["reset_state"][0]:
                state = agent.init_state(batch_size, recall=True, prev_state=state)

            # do one step of forward pass for the agent
            # output: batch_size x action_space, value: batch_size x 1
            output, value, state, mem_similarity = agent(obs, state)

            env_updated = False
            for j, o in enumerate(output):
                action_distribution = o
                action, log_prob_action, action_max = pick_action(action_distribution)
                if j == used_output_index[env_id]:
                    obs_, reward, done, _, info = env.step(list(action))
                    obs = torch.Tensor(obs_).to(device)
                    rewards.append(reward)
                    env_updated = True
                    if done.any():
                        gts.append([final_info["gt"] for final_info in info["final_info"]])
                        gt_masks.append([final_info["gt_mask"] for final_info in info["final_info"]])
                    else:
                        gts.append(info["gt"])
                        gt_masks.append(info["gt_mask"])
                outputs[j].append(o)
                probs[j].append(log_prob_action)
                actions[j].append(action)
                actions_max[j].append(action_max)
                values[j].append(value[j])
                entropys[j].append(entropy(action_distribution, device))
            assert env_updated

            mem_similarities.append(mem_similarity)
            mem_sim_entropys.append(entropy(mem_similarity, device))
            total_reward += np.sum(reward)

        # correct_actions, wrong_actions, not_know_actions = \
        #     env.compute_accuracy(torch.stack(actions[used_output_index[env_id]]))
        # actions_total_num += correct_actions + wrong_actions + not_know_actions
        # actions_correct_num += correct_actions
        # actions_wrong_num += wrong_actions
        actions_total_num += batch_size * memory_num

        if i % test_iter == 0:
            print_criterion_info = True
            print('Action distribution:', action_distribution[0])
        else:
            print_criterion_info = False

        for j in range(len(outputs)):
            probs[j] = probs[j][memory_num:]
            values[j] = values[j][memory_num:]
            entropys[j] = entropys[j][memory_num:]
            outputs[j] = torch.stack(outputs[j]).to(device)

        if criterion is not None:
            loss_all, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards[memory_num:], entropys, 
                                                                print_info=print_criterion_info, device=device)
            loss += loss_all

        if sl_criterion is not None:
            gts = torch.tensor(np.array(gts)).to(device)
            gt_masks = torch.tensor(np.array(gt_masks)).to(device)
            # print(gts, gt_masks, outputs[0].shape)
            loss_sl = sl_criterion([outputs[o][gt_masks] for o in outputs], gts[gt_masks])
            loss += loss_sl

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_max_norm, error_if_nonfinite=True)
        optimizer.step()
        loss = 0.0

        total_loss += loss_all.item()
        total_actor_loss += loss_actor.item()
        total_critic_loss += loss_critic.item()
        total_entropy += np.mean(torch.stack([torch.stack(entropys_t) for entropys_t in entropys[used_output_index[env_id]]]).cpu().detach().numpy())
            
        if i % test_iter == 0:
            if i == test_iter:
                print("Estimated time needed: {:2f}h".format((time.time()-start_time)/test_iter*num_iter/3600))
            
            # env.render()
            # gt, mask = env.get_ground_truth()
            # show example ground truth and actions, including random sampled actions and argmax actions
            # for j in range(len(outputs)):
            #     print("gt{}, action{}, max_action{}:".format(j+1, j+1, j+1), gt[0][memory_num:], 
            #         torch.stack(actions[j][memory_num:]).cpu().detach().numpy().transpose(1, 0)[0],
            #         torch.stack(actions_max[j][memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            
            accuracy = actions_correct_num / actions_total_num
            error = actions_wrong_num / actions_total_num
            not_know_rate = 1 - accuracy - error
            mean_reward = total_reward / (test_iter * batch_size)
            mean_loss = total_loss / (test_iter * batch_size)
            mean_actor_loss = total_actor_loss / (test_iter * batch_size)
            mean_critic_loss = total_critic_loss / (test_iter * batch_size)
            mean_entropy = total_entropy / (test_iter * batch_size)

            print('Iteration: {},  train accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.4f}, actor loss: {:.4f}, '
                'critic loss: {:.4f}, entropy: {:.4f}'.format(i, accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss,
                                                              mean_entropy))
            print()

            if i != 0:
                scheduler.step(-mean_reward)  # TODO: change a criterion here?
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                if lr != current_lr:
                    print("lr changed from {} to {}".format(current_lr, lr))
                    current_lr = lr

            # if error - accuracy <= min_test_loss:
            #     min_test_loss = error - accuracy
            #     save_model(agent, model_save_path, filename="model.pt")
            save_model(agent, model_save_path, filename="model.pt")
            
            # if accuracy >= stop_test_accu and i > min_iter:
            #     print("training end")
            #     break

            # test_accuracies.append(accuracy)
            # test_errors.append(error)

            total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, \
                total_actor_loss, total_critic_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
        
        if i % save_iter == 0:
            save_model(agent, model_save_path, filename="{}.pt".format(i))
    
    return test_accuracies, test_errors
