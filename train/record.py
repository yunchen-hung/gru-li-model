import torch

from models.basic_module import analyze
from .criterions.rl import pick_action


def record_model(agent, env, trials_per_condition=1, context_num=20, get_memory=False, device='cpu'):
    
    context_num = env.context_num if hasattr(env, "context_num") else context_num

    if hasattr(env, "regenerate_contexts"):
        env.regenerate_contexts()
    agent.memory_module.reset_memory()
    agent.set_retrieval(True)
    
    if get_memory:
        batch_size = env.batch_size
        memory_contexts = []
        capacity = agent.memory_module.capacity
        for batch in range(batch_size):
            actions, probs, rewards, values = [], [], [], []
            obs = torch.Tensor(env.reset()).to(device)
            done = False
            agent.set_encoding(False)
            state = agent.init_state(1)
            while not done:
                action_distribution, value, state = agent(obs, state)
                action, log_prob_action, action_max = pick_action(action_distribution)
                obs_, reward, done, info = env.step(action_max)
                obs = torch.Tensor(obs_).to(device)

                if info.get("encoding_on", False):
                    agent.set_encoding(True)
            memory_contexts.append(env.current_context_num)
            if len(memory_contexts) > capacity:
                memory_contexts.pop(0)

    data = {'actions': [], 'probs': [], 'rewards': [], 'values': [], 'readouts': [],
    'memory_contexts': []}
    actions_total_num, actions_correct_num = 0, 0
    for i in range(context_num):
        actions, probs, rewards, values, readouts, mem_contexts = [], [], [], [], [], []
        for j in range(trials_per_condition):
            if j == 0:
                obs_, info = env.reset(regenerate_contexts=True)
            else:
                obs_, info = env.reset(regenerate_contexts=False)
            obs = torch.Tensor(obs_).to(device)
            done = False
            state = agent.init_state(1)
            with analyze(agent):
                actions_trial, probs_trial, rewards_trial, values_trial = [], [], [], []
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
                        state = agent.init_state(1, recall=True)
                    
                    action_distribution, value, state = agent(obs, state)
                    action, log_prob_action, action_max = pick_action(action_distribution)
                    obs_, reward, done, info = env.step(action_max)
                    obs = torch.Tensor(obs_).to(device)

                    actions_trial.append(int(action_max))
                    probs_trial.append(log_prob_action)
                    rewards_trial.append(reward)
                    values_trial.append(value)
                actions.append(actions_trial)
                probs.append(probs_trial)
                rewards.append(rewards_trial)
                values.append(values_trial)
                readouts.append(agent.readout())
                mem_contexts.append(env.memory_sequence)
                # print(actions_trial)
                correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions_trial)
                actions_total_num += correct_actions + wrong_actions + not_know_actions
                actions_correct_num += correct_actions
        data['actions'].append(actions)
        data['probs'].append(probs)
        data['rewards'].append(rewards)
        data['values'].append(values)
        data['readouts'].append(readouts)
        data['memory_contexts'].append(mem_contexts)
    print("test accuracy: {}".format(actions_correct_num/actions_total_num))
    
    return data


# def record_model(agent, env, trials_per_condition=2, device='cpu'):
#     batch_size = env.batch_size
#     context_num = env.context_num

#     env.regenerate_contexts()
#     agent.memory_module.reset_memory()
#     agent.set_retrieval(True)
#     memory_contexts = []
#     capacity = agent.memory_module.capacity

#     for batch in range(batch_size):
#         actions, probs, rewards, values = [], [], [], []
#         obs = torch.Tensor(env.reset()).to(device)
#         done = False
#         agent.set_encoding(False)
#         state = agent.init_state(1)
#         while not done:
#             action_distribution, value, state = agent(obs, state)
#             action, log_prob_action = pick_action(action_distribution)
#             obs_, reward, done, info = env.step(action)
#             obs = torch.Tensor(obs_).to(device)

#             if info.get("encoding_on", False):
#                 agent.set_encoding(True)
#         memory_contexts.append(env.current_context_num)
#         if len(memory_contexts) > capacity:
#             memory_contexts.pop(0)

#     data = {'actions': [], 'probs': [], 'rewards': [], 'values': [], 'readouts': []}
#     for i in range(context_num):
#         actions, probs, rewards, values, readouts = [], [], [], [], []
#         for j in range(trials_per_condition):
#             obs = torch.Tensor(env.reset(context_index=i)).to(device)
#             done = False
#             agent.set_encoding(False)
#             state = agent.init_state(1)
#             with analyze(agent):
#                 actions_trial, probs_trial, rewards_trial, values_trial = [], [], [], []
#                 while not done:
#                     action_distribution, value, state = agent(obs, state)
#                     action, log_prob_action = pick_action(action_distribution)
#                     obs_, reward, done, info = env.step(action)
#                     obs = torch.Tensor(obs_).to(device)

#                     actions_trial.append(action)
#                     probs_trial.append(log_prob_action)
#                     rewards_trial.append(reward)
#                     values_trial.append(value)
#                 actions.append(actions_trial)
#                 probs.append(probs_trial)
#                 rewards.append(rewards_trial)
#                 values.append(values_trial)
#                 readouts.append(agent.readout())
#         data['actions'].append(actions)
#         data['probs'].append(probs)
#         data['rewards'].append(rewards)
#         data['values'].append(values)
#         data['readouts'].append(readouts)
    
#     data['memory_contexts'] = memory_contexts
    
#     return data
