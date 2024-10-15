import time
from collections import defaultdict
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import count_accuracy, save_model, plot_accuracy_and_error


def train(agent, envs, optimizer, scheduler, criterion, sl_criterion, ax_criterion=None,
    model_save_path=None, device='cpu', use_memory=None,
    num_iter=10000, test_iter=200, save_iter=1000, min_iter=0, stop_test_accu=1.0, 
    reset_memory=True, used_output_index=[0], env_sample_prob=[1.0],
    grad_clip=True, grad_max_norm=1.0, session_num=1):
    """
    Trains the agent using the specified environments and optimization parameters.

    Args:
        agent (object):             The agent to be trained.
        envs (list):                List of environments, each may use a different output from the agent.
        optimizer (object):         The optimizer used for updating the agent's parameters.
        scheduler (object):         The learning rate scheduler.
        criterion (object):         The loss criterion used for reinforcement learning.
        sl_criterion (object):      The loss criterion used for supervised learning.
        model_save_path (str):      Path to save the trained model.
        device (str):               Device to run the training on.
        use_memory (object):        Memory module used by the agent.
        num_iter (int):             Number of training iterations.
        test_iter (int):            Number of iterations between testing.
        save_iter (int):            Number of iterations between saving the model.
        min_iter (int):             Minimum number of iterations before stopping the training.
        stop_test_accu (float):     Accuracy threshold to stop the training.
        reset_memory (bool):        Whether to reset the memory module before each trial, used to set the agent.
        used_output_index (list):   List of indices specifying the output used for each environment.
        env_sample_prob (list, optional):   List of probabilities for sampling each environment.
        grad_clip (bool, optional):         Whether to clip the gradients during optimization.
        grad_max_norm (float, optional):    Maximum norm value for gradient clipping.

    Returns:
        test_accuracies (list): List of accuracies computed during testing.
        test_errors (list): List of errors computed during
    """
    # each used_output_index corresponds to an environment
    # specifying the output that is used for computing the action for stepping the environment
    assert len(envs) == len(used_output_index) == len(env_sample_prob)

    # set up some parameters and variables
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, \
        total_critic_loss, total_sl_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
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
        model_infos = defaultdict(list)
        rewards, mem_similarities, mem_sim_entropys = [], [], []
        gts, gt_masks = [], []
        loss_masks = []

        """ run the agent in the environment """
        correct_actions, wrong_actions, not_know_actions = 0, 0, 0

        # reset environment
        obs_, info = env.reset()
        obs = torch.Tensor(obs_).to(device)
        # print(env.memory_sequence)
        terminated = np.zeros(batch_size, dtype=bool)
        memory_num = 0
        while not terminated.all():
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
            output, value, state, model_info = agent(obs, state)

            env_updated = False
            for j, o in enumerate(output):
                action_distribution = o
                action, log_prob_action, action_max = pick_action(action_distribution)
                if j == used_output_index[env_id]:
                    # print(action)
                    obs_, reward, _, _, info = env.step(list(action))
                    done = info["done"]
                    # print(done, terminated)
                    obs = torch.Tensor(obs_).to(device)
                    rewards.append(reward)
                    env_updated = True

                    gts.append(info["gt"])
                    gt_masks.append(info["gt_mask"])
                    loss_mask = np.array(info["loss_mask"])
                    # print(info)
                    # print(loss_mask)
                    loss_masks.append(np.logical_and(loss_mask, np.logical_not(terminated)))
                    correct_actions += np.sum(info["correct"])
                    wrong_actions += np.sum(info["wrong"])
                    not_know_actions += np.sum(info["not_know"])
                    terminated = np.logical_or(terminated, done)
                outputs[j].append(o)
                probs[j].append(log_prob_action)
                actions[j].append(action)
                actions_max[j].append(action_max)
                values[j].append(value[j])
                entropys[j].append(entropy(action_distribution, device))
            assert env_updated

            for key in model_info.keys():
                model_infos[key].append(model_info[key])

            mem_similarities.append(model_info["memory_similarity"])
            mem_sim_entropys.append(entropy(model_info["memory_similarity"], device))
            total_reward += np.sum(reward)

        """ compute the loss and do backpropagation """
        if (i+1) % test_iter == 0:
            print_criterion_info = True
            print('Action distribution:', action_distribution[0])
        else:
            print_criterion_info = False

        for j in range(len(outputs)):
            probs[j] = probs[j][memory_num:]
            values[j] = values[j][memory_num:]
            entropys[j] = entropys[j][memory_num:]
            outputs[j] = torch.stack(outputs[j]).to(device)

        for key in model_infos:
            model_infos[key] = torch.stack(model_infos[key]).to(device)

        # compute RL loss
        if criterion is not None:
            loss_rl, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards[memory_num:], entropys, loss_masks[memory_num:],
                                                                print_info=print_criterion_info, device=device)
            loss += loss_rl

            total_loss += loss_rl.item()
            total_actor_loss += loss_actor.item()
            total_critic_loss += loss_critic.item()
            total_entropy += np.mean(torch.stack([torch.stack(entropys_t) for entropys_t \
                                                  in entropys[used_output_index[env_id]]]).cpu().detach().numpy())

        # compute SL loss
        gts = torch.tensor(np.array(gts)).to(device)    # time x batch_size
        gt_masks = torch.tensor(np.array(gt_masks)).to(device)
        # print(gts, gt_masks)
        if sl_criterion is not None:
            # print(gts, gt_masks, outputs[0].shape)
            loss_sl = sl_criterion([outputs[o][gt_masks] for o in outputs], gts[gt_masks])
            loss += loss_sl
            total_sl_loss += loss_sl.item()

        # compute auxiliary loss
        if ax_criterion is not None:
            loss_ax = ax_criterion(device=device, **model_infos)
            loss += loss_ax
            if print_criterion_info:
                print("auxiliary loss:", loss_ax.item())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_max_norm, error_if_nonfinite=True)
        optimizer.step()
        loss = 0.0

        # compute accuracy
        # correct_actions, wrong_actions, not_know_actions = \
        #     compute_accuracy(torch.stack(actions[used_output_index[env_id]]), gts, gt_masks)
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions
        # actions_total_num += batch_size * memory_num

        """ print training information and save model """
        if (i+1) % test_iter == 0:
            if (i+1) == test_iter:
                print("Estimated time needed: {:2f}h".format((time.time()-start_time)/test_iter*num_iter/3600))
            
            accuracy = actions_correct_num / actions_total_num
            error = actions_wrong_num / actions_total_num
            not_know_rate = 1 - accuracy - error
            mean_reward = total_reward / (test_iter * batch_size)
            mean_loss = total_loss / (test_iter * batch_size)
            mean_actor_loss = total_actor_loss / (test_iter * batch_size)
            mean_critic_loss = total_critic_loss / (test_iter * batch_size)
            mean_entropy = total_entropy / (test_iter * batch_size)

            print('Iteration: {},  train accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.4f}, actor loss: {:.4f}, '
                'critic loss: {:.4f}, entropy: {:.4f}'.format(i+1, accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss,
                                                              mean_entropy))
            actions_trial = torch.stack(actions[used_output_index[env_id]]).cpu().detach().numpy()
            gts_trial = gts.cpu().detach().numpy()
            # print(actions_trial.shape, gts_trial.shape)
            if sl_criterion is not None:
                print("encoding phase, action:", actions_trial[0:memory_num, 0], "gt:", gts_trial[0:memory_num, 0])
            if criterion is not None:
                # print(gts_trial[0:memory_num, 0])
                # print(actions_trial, gts_trial, loss_masks)
                print("recall phase, action:", actions_trial[memory_num:, 0], "gt:", gts_trial[memory_num:, 0])
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
            
            if accuracy >= stop_test_accu and i > min_iter:
                print("training end")
                break

            test_accuracies.append(accuracy)
            test_errors.append(error)

            total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, \
                total_actor_loss, total_critic_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
            
            plot_accuracy_and_error(test_accuracies, test_errors, model_save_path, filename="accuracy_session_{}.png".format(session_num))
        
        if i+1 % save_iter == 0:
            save_model(agent, model_save_path, filename="{}.pt".format(i))
    
    return test_accuracies, test_errors
