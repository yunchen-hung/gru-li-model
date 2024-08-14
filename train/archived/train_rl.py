import time
from collections import defaultdict
import numpy as np
import torch

from train.criterions.rl import pick_action
from models.utils import entropy
from train.utils import count_accuracy, save_model
from torch.nn.functional import mse_loss


def train_model(agent, envs, optimizer, scheduler, setup, criterion, sl_criterion=None, test=False, 
    model_save_path=None, device='cpu', use_memory=None,
    num_iter=10000, test_iter=200, save_iter=1000, min_iter=0, step_iter=1, batch_size=1, stop_test_accu=1.0, 
    mem_beta_decay_rate=1.0, mem_beta_decay_acc=1.0, mem_beta_min=0.01, 
    randomly_flush_state=False, flush_state_prob=1.0, use_memory_together_with_flush=False, reset_memory=True,
    memory_entropy_reg=False, memory_reg_weight=0.0, sl_criterion_weight=1.0,
    used_output_index=[0], env_sample_prob=[1.0],
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
    batch_size = batch_size
    if use_memory:
        agent.use_memory = use_memory
    min_test_loss = torch.inf

    print("start training")
    print("batch size:", batch_size)

    current_step_iter = 0

    forward_time, backward_time = 0.0, 0.0
    loss_time = 0.0

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

        # if randomly_flush_state is true, randomly decide whether to flush the state between encoding and recall phase
        # if randomly_use_memory is also true, don't use memory when don't flush state
        # this is used for mixed task of working memory and episodic memory
        if randomly_flush_state:
            if np.random.rand() < flush_state_prob:
                env.reset_state_before_test = True
                if use_memory_together_with_flush:
                    agent.use_memory = True
            else:
                env.reset_state_before_test = False
                if use_memory_together_with_flush:
                    agent.use_memory = False

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

        forward_start_time = time.time()

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
                # prev_state = state

            # do one step of forward pass for the agent
            # output: batch_size x action_space, value: batch_size x 1
            output, value, state, mem_similarity = agent(obs, state)

            env_updated = False
            for j, o in enumerate(output):
                action_distribution = o
                action, log_prob_action, action_max = pick_action(action_distribution)
                if j == used_output_index[env_id]:
                    obs_, reward, done, info = env.step(action)
                    obs = torch.Tensor(obs_).to(device)
                    rewards.append(reward)
                    env_updated = True
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

        correct_actions, wrong_actions, not_know_actions = \
            env.compute_accuracy(torch.stack(actions[used_output_index[env_id]]))
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        # forward_time += time.time() - forward_start_time

        # loss_start_time = time.time()

        if i % test_iter == 0:
            print_criterion_info = True
            print('Action distribution:', action_distribution[0])
        else:
            print_criterion_info = False

        for j in range(len(outputs)):
            probs[j] = probs[j][memory_num:]
            values[j] = values[j][memory_num:]
            entropys[j] = entropys[j][memory_num:]

        # print(probs)
        # print(values)
        # print(rewards[memory_num:])
        # print(entropys)
        loss_all, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards[memory_num:], entropys, 
                                                                print_info=print_criterion_info, device=device)
        loss += loss_all
        # print(loss)

        if memory_entropy_reg:
            # add (negative) entropy regularization for memory similarity
            # to encourage the memory similarity to be closer to one-hot
            mem_ent_reg_loss = memory_reg_weight * torch.mean(torch.stack([torch.stack(mem_sim_ent) for mem_sim_ent in mem_sim_entropys[memory_num:]]))
            loss += mem_ent_reg_loss
        else:
            mem_ent_reg_loss = None

        if sl_criterion is not None:
            gt, mask = env.get_ground_truth(phase="encoding")
            gt = torch.tensor(gt).to(device)
            mask = torch.tensor(mask).to(device)
            outputs_sl = defaultdict(list)
            for j in range(len(outputs)):
                outputs_sl[j] = torch.stack(outputs[j])[mask.reshape(-1, 1) == 1]
            loss += sl_criterion(outputs_sl, gt[mask==1].reshape(-1, 1)) * sl_criterion_weight

        # loss_time += time.time() - loss_start_time

        # backward_start_time = time.time()

        # print(loss)
        # print()

        current_step_iter += 1
        if current_step_iter == step_iter:
            loss /= step_iter
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_max_norm, error_if_nonfinite=True)
            optimizer.step()
            current_step_iter = 0
            loss = 0.0

        # backward_time += time.time() - backward_start_time

        total_loss += loss_all.item()
        total_actor_loss += loss_actor.item()
        total_critic_loss += loss_critic.item()
        # print(entropys)
        total_entropy += np.mean(torch.stack([torch.stack(entropys_t) for entropys_t in entropys[used_output_index[env_id]]]).cpu().detach().numpy())
            
        if i % test_iter == 0:
            if i == test_iter:
                print("Estimated time needed: {:2f}h".format((time.time()-start_time)/test_iter*num_iter/3600))
            
            # print("Forward time: {:.2f}s, Loss time: {:.2f}s, Backward time: {:.2f}s".format(forward_time, loss_time, backward_time))

            env.render()
            gt, mask = env.get_ground_truth()
            # show example ground truth and actions, including random sampled actions and argmax actions
            # print(torch.tensor(actions[memory_num:]).shape)
            # print(torch.tensor(actions[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            # print(torch.tensor(actions_max[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            for j in range(len(outputs)):
                print("gt{}, action{}, max_action{}:".format(j+1, j+1, j+1), gt[0][memory_num:], 
                    torch.stack(actions[j][memory_num:]).cpu().detach().numpy().transpose(1, 0)[0],
                    torch.stack(actions_max[j][memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            # print("gt, actions, max_actions:", gt[0][memory_num:], torch.stack(actions[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0], 
            #     torch.stack(actions_max[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            
            # show example in the encoding phase
            # gt_encoding = env.get_ground_truth(phase="encoding")
            # print("encoding gt, actions, max_actions:", gt[0][:memory_num], torch.stack(actions[:memory_num]).cpu().detach().numpy().transpose(1, 0)[0], 
            #     torch.stack(actions_max[:memory_num]).cpu().detach().numpy().transpose(1, 0)[0])

            # print("Actor loss, Critic loss, Entropy loss, Mem entropy loss:", loss_actor.item(), loss_critic.item(), loss_ent_reg.item(), mem_ent_reg_loss.item() if memory_entropy_reg else 0.0)

            accuracy = actions_correct_num / actions_total_num
            error = actions_wrong_num / actions_total_num
            not_know_rate = 1 - accuracy - error
            mean_reward = total_reward / actions_total_num
            mean_loss = total_loss / test_iter
            mean_actor_loss = total_actor_loss / test_iter
            mean_critic_loss = total_critic_loss / test_iter
            mean_entropy = total_entropy / test_iter

            if agent.mem_beta is not None:
                print('mem_beta:', agent.mem_beta)
            
            # print('variance of hidden state:', torch.var(prev_state).item())

            print('Iteration: {},  train accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.4f}, actor loss: {:.4f}, '
                'critic loss: {:.4f}, entropy: {:.4f}'.format(i, accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss,
                                                              mean_entropy))

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
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                if lr != current_lr:
                    print("lr changed from {} to {}".format(current_lr, lr))
                    current_lr = lr

            if test_error - test_accuracy <= min_test_loss:
                min_test_loss = test_error - test_accuracy
                save_model(agent, model_save_path, filename="model.pt")
            
            if test_accuracy >= stop_test_accu and i > min_iter:
                print("training end")
                break  

            if agent.mem_beta is not None and test_accuracy >= mem_beta_decay_acc and agent.mem_beta >= mem_beta_min:
                agent.mem_beta *= mem_beta_decay_rate

            test_accuracies.append(test_accuracy)
            test_errors.append(test_error)

            total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, total_critic_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
        
        if i % save_iter == 0:
            save_model(agent, model_save_path, filename="{}.pt".format(i))
    
    return test_accuracies, test_errors

