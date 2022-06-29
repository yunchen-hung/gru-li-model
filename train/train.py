import torch

from models.rl import pick_action, compute_returns, compute_a2c_loss
from models.utils import entropy
from .utils import count_accuracy, save_model
from utils import import_attr


# TODO: support other RL algorithms
def train_model(agent, env, optimizer, scheduler, setup, num_iter=10000, test=True, test_iter=200, save_iter=1000, stop_test_accu=1.0, device='cpu', model_save_path=None):
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, total_critic_loss = 0.0, 0, 0, 0, 0.0, 0.0, 0.0
    test_accuracies = []
    test_errors = []

    keep_state = setup.get("keep_state", False)    # reset state after each episode
    if keep_state:
        print("keep state")
    regenerate_context = setup.get("regenerate_context", True)    # regenerate context after each episode
    loss_setup = setup.get("loss_setup", {})

    batch_size = env.batch_size

    print("start training")
    print("batch size:", batch_size)
    # state = agent.init_state(1)  # TODO: possibly add batch size here
    min_test_loss = torch.inf
    for i in range(num_iter):
        state = agent.init_state(1)  # TODO: possibly add batch size here
        if regenerate_context and hasattr(env, "regenerate_contexts"):
            env.regenerate_contexts()
        agent.memory_module.reset_memory()
        agent.set_retrieval(True)
        for batch in range(batch_size):
            actions, probs, rewards, values, entropys = [], [], [], [], []

            if keep_state:
                new_state = []
                for item in state:
                    new_state.append(item.detach().clone())
                state = tuple(new_state)
            else:
                state = agent.init_state(1)

            obs_, info = env.reset()
            obs = torch.Tensor(obs_).to(device)
            done = False
            while not done:
                if info.get("encoding_on", False):
                    agent.set_encoding(True)
                else:
                    agent.set_encoding(False)
                if info.get("retrieval_off", False):
                    agent.set_retrieval(False)
                else:
                    agent.set_retrieval(True)
                if info.get("reset_state", False):
                    state = agent.init_state(1)
                # print(agent.memory_module.stored_memory)

                # torch.autograd.set_detect_anomaly(True)
                action_distribution, value, state = agent(obs, state)
                action, log_prob_action = pick_action(action_distribution)
                obs_, reward, done, info = env.step(action)
                obs = torch.Tensor(obs_).to(device)

                probs.append(log_prob_action)
                rewards.append(reward)
                values.append(value)
                entropys.append(entropy(action_distribution))
                actions.append(action)
                total_reward += reward

            correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions)
            # print(torch.stack(actions[env.memory_num:]).detach().cpu().numpy(), env.memory_sequence, correct_actions, wrong_actions, not_know_actions)
            actions_total_num += correct_actions + wrong_actions + not_know_actions
            actions_correct_num += correct_actions
            actions_wrong_num += wrong_actions

            loss, loss_actor, loss_critic = compute_a2c_loss(probs, values, rewards, entropys, device=device, **loss_setup)

            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            optimizer.step()

            total_loss += loss.item()
            total_actor_loss += loss_actor.item()
            total_critic_loss += loss_critic.item()
            
        if i % test_iter == 0:
            accuracy = actions_correct_num / actions_total_num
            error = actions_wrong_num / actions_total_num
            not_know_rate = 1 - accuracy - error
            mean_reward = total_reward / actions_total_num
            mean_loss = total_loss / test_iter
            mean_actor_loss = total_actor_loss / test_iter
            mean_critic_loss = total_critic_loss / test_iter

            print('Iteration: {},  train accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.2f}, actor loss: {:.2f}, '
                'critic loss: {:.2f}'.format(i, accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss))

            if test:
                test_accuracy, test_error, test_not_know_rate, test_mean_reward, test_mean_loss, test_mean_actor_loss, test_mean_critic_loss \
                    = count_accuracy(agent, env, num_trials_per_condition=10, device=device)

                print('\ttest accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.2f}, actor loss: {:.2f}, critic loss: {:.2f}'\
                    .format(test_accuracy, test_error, test_not_know_rate, test_mean_reward, test_mean_loss, test_mean_actor_loss, test_mean_critic_loss))
            else:
                test_error = error
                test_accuracy = accuracy
                test_mean_loss = mean_loss

            scheduler.step(test_error - test_accuracy)  # TODO: change a criterion here?

            if test_error - test_accuracy < min_test_loss:
                min_test_loss = test_error - test_accuracy
                save_model(agent, model_save_path, filename="model.pt")
            
            if test_accuracy >= stop_test_accu:
                break

            test_accuracies.append(test_accuracy)
            test_errors.append(test_error)

            total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, total_critic_loss = 0.0, 0, 0, 0, 0.0, 0.0, 0.0
        
        if i % save_iter == 0:
            save_model(agent, model_save_path, filename="{}.pt".format(i))
    
    return test_accuracies, test_errors


def supervised_train_model(agent, env, optimizer, scheduler, setup, criterion="CrossEntropyLoss", num_iter=10000, test=False, test_iter=200, save_iter=1000, stop_test_accu=1.0, device='cpu', model_save_path=None):
    actions_correct_num, actions_wrong_num, actions_total_num, total_loss = 0, 0, 0, 0.0
    test_accuracies = []
    test_errors = []

    criterion = import_attr("torch.nn.{}".format(criterion))()

    keep_state = setup.get("keep_state", False)    # reset state after each episode
    if keep_state:
        print("keep state")

    batch_size = env.batch_size

    print("start supervised training")
    print("batch size:", batch_size)
    # state = agent.init_state(1)  # TODO: possibly add batch size here
    min_test_loss = torch.inf
    for i in range(num_iter):
        state = agent.init_state(1)  # TODO: possibly add batch size here
        agent.memory_module.reset_memory()
        agent.set_encoding(False)
        agent.set_retrieval(False)
        for batch in range(batch_size):
            env.reset()
            data, gt = env.get_batch()
            data = torch.as_tensor(data, dtype=torch.float).to(device)
            # gt = torch.as_tensor(gt, dtype=torch.long).to(device)
            gt = torch.as_tensor(gt, dtype=torch.float).to(device)
            actions, values = [], []

            if keep_state:
                new_state = []
                for item in state:
                    new_state.append(item.detach().clone())
                state = tuple(new_state)
            else:
                state = agent.init_state(1)

            outputs = []
            for t in range(data.shape[0]):
                # if info.get("encoding_on", False):
                #     agent.set_encoding(True)
                # else:
                #     agent.set_encoding(False)
                # if info.get("retrieval_off", False):
                #     agent.set_retrieval(False)
                # else:
                #     agent.set_retrieval(True)
                # if info.get("reset_state", False):
                #     state = agent.init_state(1)

                # torch.autograd.set_detect_anomaly(True)
                output, value, state = agent(data[t], state)

                values.append(value)
                actions.append(torch.argmax(output).item())
                outputs.append(output)
            outputs = torch.stack(outputs)

            correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions)
            # print(torch.stack(actions[env.memory_num:]).detach().cpu().numpy(), env.memory_sequence, correct_actions, wrong_actions, not_know_actions)
            actions_total_num += correct_actions + wrong_actions + not_know_actions
            actions_correct_num += correct_actions
            actions_wrong_num += wrong_actions

            # print(outputs[env.memory_num:].shape, gt[env.memory_num:].shape)
            # print(outputs, gt)
            # print(actions, gt)
            loss = criterion(outputs[env.memory_num:], gt[env.memory_num:])  # TODO: add an attr in env to specify how long output to use for loss

            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        if i % test_iter == 0:
            # print(actions, gt)
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

            scheduler.step(test_error - test_accuracy)  # TODO: change a criterion here?

            if test_error - test_accuracy < min_test_loss:
                min_test_loss = test_error - test_accuracy
                save_model(agent, model_save_path, filename="model.pt")
            
            if test_accuracy >= stop_test_accu:
                break

            test_accuracies.append(test_accuracy)
            test_errors.append(test_error)

            actions_correct_num, actions_wrong_num, actions_total_num, total_loss = 0, 0, 0, 0.0
        
        if i % save_iter == 0:
            save_model(agent, model_save_path, filename="sup_{}.pt".format(i))
    
    return test_accuracies, test_errors
