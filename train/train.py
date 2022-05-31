import torch

from models.rl import pick_action, compute_returns, compute_a2c_loss
from models.utils import entropy
from .utils import count_accuracy, save_model


# TODO: support other RL algorithms
def train_model(agent, env, optimizer, scheduler, setup, num_iter=10000, test_iter=200, save_iter=1000, stop_test_accu=1.0, device='cpu', model_save_path=None):
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, total_critic_loss = 0.0, 0, 0, 0, 0.0, 0.0, 0.0
    test_accuracies = []
    test_errors = []

    batch_size = env.batch_size
    
    print("start training")
    print("batch size:", batch_size)
    for i in range(num_iter):
        env.regenerate_contexts()
        agent.memory_module.reset_memory()
        agent.set_retrieval(True)
        for batch in range(batch_size):
            actions, probs, rewards, values, entropys = [], [], [], [], []

            obs = torch.Tensor(env.reset()).to(device)
            done = False
            state = agent.init_state(1)  # TODO: possibly add batch size here
            agent.set_encoding(False)
            while not done:
                torch.autograd.set_detect_anomaly(True)
                action_distribution, value, state = agent(obs, state)
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

            actions_total_num += len(actions)
            correct_actions, wrong_actions = env.compute_accuracy(actions)
            actions_correct_num += correct_actions
            actions_wrong_num += wrong_actions

            returns = compute_returns(rewards, normalize=True)  # TODO: make normalize a parameter
            loss_actor, loss_critic = compute_a2c_loss(probs, values, returns, device=device)
        
            pi_ent = torch.stack(entropys).sum()
            loss = loss_actor + loss_critic - pi_ent * 0.1  # 0.1: eta, make it a parameter

            total_loss += loss.item()
            total_actor_loss += loss_actor.item()
            total_critic_loss += loss_critic.item()

            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            optimizer.step()
            
        min_test_loss = torch.inf
        if i % test_iter == 0:
            accuracy = actions_correct_num / actions_total_num
            error = actions_wrong_num / actions_total_num
            not_know_rate = 1 - accuracy - error
            mean_reward = total_reward / actions_total_num
            mean_loss = total_loss / test_iter
            mean_actor_loss = total_actor_loss / test_iter
            mean_critic_loss = total_critic_loss / test_iter

            test_accuracy, test_error, test_not_know_rate, test_mean_reward, test_mean_loss, test_mean_actor_loss, test_mean_critic_loss \
                = count_accuracy(agent, env, num_trials_per_condition=10, device=device)

            scheduler.step(test_error - test_accuracy)  # TODO: change a criterion here?

            print('Iteration: {},  train accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.2f}, actor loss: {:.2f}, '
                'critic loss: {:.2f}'.format(i, accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss))
            print('\ttest accuracy: {:.2f}, error: {:.2f}, no action: {:.2f}, mean reward: {:.2f}, total loss: {:.2f}, actor loss: {:.2f}, critic loss: {:.2f}'\
                .format(test_accuracy, test_error, test_not_know_rate, test_mean_reward, test_mean_loss, test_mean_actor_loss, test_mean_critic_loss))

            if test_mean_loss < min_test_loss:
                min_test_loss = test_mean_loss
                save_model(agent, model_save_path, filename="model.pt")
            
            if test_accuracy >= stop_test_accu:
                break

            test_accuracies.append(test_accuracy)
            test_errors.append(test_error)

            total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, total_critic_loss = 0.0, 0, 0, 0, 0.0, 0.0, 0.0
        
        if i % save_iter == 0:
            save_model(agent, model_save_path, filename="{}.pt".format(i))
    
    return test_accuracies, test_errors
