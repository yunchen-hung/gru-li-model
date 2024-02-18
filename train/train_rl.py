import time
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import count_accuracy, save_model
from torch.nn.functional import mse_loss


def train_model(agent, env, optimizer, scheduler, setup, criterion, sl_criterion=None, test=False, stop_test_accu=1.0, model_save_path=None, device='cpu',
    num_iter=10000, test_iter=200, save_iter=1000, min_iter=0, step_iter=1, batch_size=1, use_memory=None,
    mem_beta_decay_rate=1.0, mem_beta_decay_acc=1.0, mem_beta_min=0.01, 
    randomly_flush_state=False, flush_state_prob=1.0, use_memory_together_with_flush=False, 
    memory_entropy_reg=False, memory_reg_weight=0.0):
    """
    Train the model with RL
    """
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
        agent.reset_memory()

        # create variables to store data related to outputs and results
        actions, probs, rewards, values, entropys, actions_max, outputs, mem_similarities, mem_sim_entropys = \
            [], [], [], [], [], [], [], [], []
        outputs2 = []

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

            # do one step of forward pass for the agent
            # output: batch_size x action_space, value: batch_size x 1
            output, value, output2, value2, state, mem_similarity = agent(obs, state)
            # action_distribution: batch_size x action_space
            if isinstance(output, tuple):
                # when generating two decisions, only record the first one as action
                action_distribution = output[0]
            else:
                action_distribution = output
            # action: batch_size, log_prob_action: batch_size
            action, log_prob_action, action_max = pick_action(action_distribution)
            # info_ = info
            obs_, reward, done, info = env.step(action)
            # print(obs, action, reward, info_)
            obs = torch.Tensor(obs_).to(device)

            outputs.append(output)
            outputs2.append(output2)
            probs.append(log_prob_action)
            rewards.append(reward)
            values.append(value)
            entropys.append(entropy(action_distribution, device))
            actions.append(action)
            actions_max.append(action_max)
            mem_similarities.append(mem_similarity)
            mem_sim_entropys.append(entropy(mem_similarity, device))
            total_reward += np.sum(reward)

        # print(actions)
        # print(np.array([action.item() for action in actions]))
        # correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(np.array([action.item() for action in actions]))
        # print(torch.stack(actions).shape)
        correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(torch.stack(actions))
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        forward_time += time.time() - forward_start_time

        loss_start_time = time.time()

        if i % test_iter == 0:
            print_criterion_info = True
            print('Action distribution:', action_distribution[0])
        else:
            print_criterion_info = False

        loss, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards, entropys, memory_num=memory_num, print_info=print_criterion_info, device=device)

        if memory_entropy_reg:
            # add (negative) entropy regularization for memory similarity
            # to encourage the memory similarity to be closer to one-hot
            # print([len(mem_sim_ent) for mem_sim_ent in mem_sim_entropys])
            mem_ent_reg_loss = memory_reg_weight * torch.mean(torch.stack([torch.stack(mem_sim_ent) for mem_sim_ent in mem_sim_entropys[memory_num:]]))
            loss += mem_ent_reg_loss
        else:
            mem_ent_reg_loss = None

        if sl_criterion is not None:
            gt = torch.tensor(env.get_ground_truth(phase="encoding")).to(device)
            if outputs2[0] is not None:
                # print(outputs[:memory_num], gt.T)
                outputs2 = torch.stack(outputs2)
                loss += sl_criterion(outputs2[:memory_num], gt.T, memory_num=memory_num)
            else:
                outputs = torch.stack(outputs)
                loss += sl_criterion(outputs[:memory_num], gt.T, memory_num=memory_num)

        loss_time += time.time() - loss_start_time

        backward_start_time = time.time()

        current_step_iter += 1
        if current_step_iter == step_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_step_iter = 0

        backward_time += time.time() - backward_start_time

        total_loss += loss.item()
        total_actor_loss += loss_actor.item()
        total_critic_loss += loss_critic.item()
        total_entropy += np.mean(torch.stack([torch.stack(entropys_t) for entropys_t in entropys]).cpu().detach().numpy())
            
        if i % test_iter == 0:
            if i == test_iter:
                print("Estimated time needed: {:2f}h".format((time.time()-start_time)/test_iter*num_iter/3600))
            
            print("Forward time: {:.2f}s, Loss time: {:.2f}s, Backward time: {:.2f}s".format(forward_time, loss_time, backward_time))

            env.render()
            gt = env.get_ground_truth()
            # show example ground truth and actions, including random sampled actions and argmax actions
            # print(torch.tensor(actions[memory_num:]).shape)
            # print(torch.tensor(actions[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            # print(torch.tensor(actions_max[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            print("gt, actions, max_actions:", gt[0], torch.stack(actions[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0], 
                torch.stack(actions_max[memory_num:]).cpu().detach().numpy().transpose(1, 0)[0])
            # show example in the encoding phase
            gt_encoding = env.get_ground_truth(phase="encoding")
            print("encoding gt, actions, max_actions:", gt_encoding[0], torch.stack(actions[:memory_num]).cpu().detach().numpy().transpose(1, 0)[0], 
                torch.stack(actions_max[:memory_num]).cpu().detach().numpy().transpose(1, 0)[0])
            if sl_criterion is not None and outputs2[0] is not None:
                print("gt2, encoding action2:", gt[0], torch.argmax(outputs2[:memory_num, 0], dim=1).detach().cpu().numpy().reshape(-1))

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

