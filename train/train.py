import time
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import count_accuracy, save_model
from torch.nn.functional import mse_loss


def train_model(agent, env, optimizer, scheduler, setup, criterion, num_iter=10000, test=False, test_iter=200, save_iter=1000, stop_test_accu=1.0, device='cpu', 
    model_save_path=None, use_memory=None, train_all_time=False, train_encode=False, train_encode_2item=False, train_encode_weight=1.0, min_iter=0, step_iter=1,
    mem_beta_decay_rate=1.0, mem_beta_decay_acc=1.0, mem_beta_min=0.01, randomly_flush_state=False, flush_state_prob=1.0, randomly_use_memory=False, 
    memory_entropy_reg=False, memory_reg_weight=0.0):
    """
    Train the model with RL
    """
    # set up some parameters and variables
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, \
        total_critic_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
    test_accuracies, test_errors = [], []
    batch_size = env.batch_size
    if use_memory:
        agent.use_memory = use_memory
    min_test_loss = torch.inf

    print("start training")
    print("batch size:", batch_size)

    current_step_iter = 0
    
    for i in range(num_iter):
        # record time for the first iteration to estimate total time needed
        if i == 0:
            start_time = time.time()

        # if randomly_flush_state is true, randomly decide whether to flush the state between encoding and recall phase
        # if randomly_use_memory is also true, don't use memory when don't flush state
        # this is used for mixed task of working memory and episodic memory
        if randomly_flush_state:
            if np.random.rand() < flush_state_prob:
                env.reset_state_before_test = True
                if randomly_use_memory:
                    agent.use_memory = True
            else:
                env.reset_state_before_test = False
                if randomly_use_memory:
                    agent.use_memory = False

        # before each trial, for the agent:
        # 1. reset initial state
        # 2. reset memory module
        state = agent.init_state(batch_size)
        agent.reset_memory()

        # create variables to store data related to outputs and results
        actions, probs, rewards, values, entropys, actions_max, outputs, mem_similarities, mem_sim_entropys = \
            [], [], [], [], [], [], [], [], []

        # reset environment
        obs_, info = env.reset()
        obs = torch.Tensor(obs_).to(device)
        # print(env.memory_sequence)
        done = False
        while not done:
            # set up the phase of the agent
            if info["phase"] == "encoding":
                agent.set_encoding(True)
                agent.set_retrieval(False)
            elif info["phase"] == "recall":
                agent.set_encoding(False)
                agent.set_retrieval(True)
            # reset state between phases
            if info.get("reset_state", False):
                state = agent.init_state(batch_size, recall=True, prev_state=state)

            # do one step of forward pass for the agent
            output, value, state, mem_similarity = agent(obs, state)
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

            outputs.append(output)
            probs.append(log_prob_action)
            rewards.append(reward)
            values.append(value)
            entropys.append(entropy(action_distribution, device))
            actions.append(action)
            actions_max.append(action_max)
            mem_similarities.append(mem_similarity)
            mem_sim_entropys.append(entropy(mem_similarity, device))
            total_reward += np.sum(reward)
        
        # print()

        correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions)
        # print(torch.stack(actions[env.memory_num:]).detach().cpu().numpy(), env.memory_sequence, correct_actions, wrong_actions, not_know_actions)
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        if i % test_iter == 0:
            print_criterion_info = True
            print(action_distribution)
        else:
            print_criterion_info = False

        if train_all_time:
            # train both encoding and recall phase
            loss, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards, entropys, print_info=print_criterion_info, device=device)
        else:
            loss, loss_actor, loss_critic, loss_ent_reg = criterion(probs[env.current_memory_num:], values[env.current_memory_num:], rewards[env.current_memory_num:], entropys[env.current_memory_num:], print_info=print_criterion_info, device=device)

        if train_encode:
            # train encoding phase with supervised loss
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
                outputs = (outputs,)
            _, gt = env.get_batch()
            gt = torch.as_tensor(gt, dtype=torch.float).to(device)
            loss += train_encode_weight * mse_loss(outputs[0][:env.current_memory_num], gt[:env.current_memory_num])
            if train_encode_2item:
                loss += train_encode_weight * mse_loss(outputs[1][1:env.current_memory_num], gt[:env.current_memory_num-1])

        if memory_entropy_reg:
            # add (negative) entropy regularization for memory similarity
            # to encourage the memory similarity to be closer to one-hot
            mem_ent_reg_loss = memory_reg_weight * torch.mean(torch.stack([torch.stack(mem_sim_ent) for mem_sim_ent in mem_sim_entropys]))
            loss += mem_ent_reg_loss
        else:
            mem_ent_reg_loss = None

        current_step_iter += 1
        if current_step_iter == step_iter:
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            optimizer.step()
            current_step_iter = 0

        total_loss += loss.item()
        total_actor_loss += loss_actor.item()
        total_critic_loss += loss_critic.item()
        total_entropy += np.mean(torch.stack([torch.stack(entropys_t) for entropys_t in entropys]).cpu().detach().numpy())
            
        if i % test_iter == 0:
            if i == test_iter:
                print("Estimated time needed: {:2f}h".format((time.time()-start_time)/test_iter*num_iter/3600))

            gt = env.memory_sequence
            # show example ground truth and actions, including random sampled actions and argmax actions
            print(gt, torch.tensor(actions[env.current_memory_num:]).cpu().detach().numpy().transpose(1, 0)[0], 
                torch.tensor(actions_max[env.current_memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            if train_all_time or train_encode:
                # show example in the encoding phase
                print(torch.tensor(actions[:env.current_memory_num]).cpu().detach().numpy().transpose(1, 0)[0], 
                    torch.tensor(actions_max[:env.current_memory_num]).cpu().detach().numpy().transpose(1, 0)[0])

            print("Actor loss, Critic loss, Entropy loss, Mem entropy loss:", loss_actor.item(), loss_critic.item(), loss_ent_reg.item(), mem_ent_reg_loss.item() if memory_entropy_reg else 0.0)

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

            if i != 0:
                scheduler.step(test_error - test_accuracy)  # TODO: change a criterion here?

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


def supervised_train_model(agent, env, optimizer, scheduler, setup, criterion, num_iter=10000, test=False, test_iter=200, save_iter=1000, stop_test_accu=1.0, 
    train_all_time=False, device='cpu', model_save_path=None, use_memory=None, min_iter=0, random_action=False):
    actions_correct_num, actions_wrong_num, actions_total_num, total_loss = 0, 0, 0, 0.0
    test_accuracies = []
    test_errors = []

    batch_size = env.batch_size

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
        obs_, info = env.reset()
        obs = torch.Tensor(obs_).to(device)
        # print(env.memory_sequence)
        done = False

        data, gt = env.get_batch()
        data = torch.as_tensor(data, dtype=torch.float).to(device)
        gt = torch.as_tensor(gt, dtype=torch.float).to(device)

        while not done:
            # set up the phase of the agent
            if info["phase"] == "encoding":
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

        if random_action:
            correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions)
        else:
            correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions_max)
        # rewards = env.compute_rewards(actions)
        # print(torch.stack(actions[env.memory_num:]).detach().cpu().numpy(), env.memory_sequence, correct_actions, wrong_actions, not_know_actions)
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        # print(outputs[env.memory_num:].shape, gt[env.memory_num:].shape)
        # print(outputs, gt)
        # print(actions, gt)
        # loss = criterion(outputs[env.memory_num:], gt[env.memory_num:])  # TODO: add an attr in env to specify how long output to use for loss4
        if isinstance(outputs, tuple):
            if train_all_time:
                outputs_list = [output[env.current_memory_num:] for output in outputs]
                outputs_list.extend([output[:env.current_memory_num] for output in outputs])
                loss = criterion(tuple(outputs_list), gt[env.current_memory_num:])
            else:
                loss = criterion(tuple([output[env.current_memory_num:] for output in outputs]), gt[env.current_memory_num:])
        else:
            if train_all_time:
                loss = criterion(tuple([outputs[env.current_memory_num:], outputs[:env.current_memory_num]]), gt[env.current_memory_num:])
            else:
                loss = criterion(outputs[env.current_memory_num:], gt[env.current_memory_num:])
                
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % test_iter == 0:
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    print(env.memory_sequence[0], np.array(actions)[env.current_memory_num:,0],
                        list(torch.argmax(outputs[0][:env.current_memory_num], dim=2).detach().cpu().numpy().reshape(-1)),
                        list(torch.argmax(outputs[1][env.current_memory_num:], dim=2).detach().cpu().numpy().reshape(-1)),
                        list(torch.argmax(outputs[2][:env.current_memory_num], dim=2).detach().cpu().numpy().reshape(-1)))
                else:
                    print(env.memory_sequence[0], np.array(actions)[env.current_memory_num:,0], 
                        list(torch.argmax(outputs[0][:env.current_memory_num], dim=2).detach().cpu().numpy().reshape(-1)),
                        list(torch.argmax(outputs[1], dim=2).detach().cpu().numpy().reshape(-1)))
            else:
                print(env.memory_sequence[0], np.array(actions)[env.current_memory_num:,0], 
                    torch.argmax(outputs[:env.current_memory_num], dim=2).detach().cpu().numpy().reshape(-1))
            # print(outputs)
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
