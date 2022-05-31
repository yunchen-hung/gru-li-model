import torch

from models.basic_module import analyze
from models.rl import pick_action, compute_returns, compute_a2c_loss
from models.utils import entropy
from .utils import count_accuracy, save_model


def record_model(agent, env, trials_per_condition=2, device='cpu'):
    batch_size = env.batch_size
    context_num = env.context_num

    env.regenerate_contexts()
    agent.memory_module.reset_memory()
    agent.set_retrieval(True)
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
            action, log_prob_action = pick_action(action_distribution)
            obs_, reward, done, info = env.step(action)
            obs = torch.Tensor(obs_).to(device)

            if info.get("encoding_on", False):
                agent.set_encoding(True)
        memory_contexts.append(env.current_context_num)
        if len(memory_contexts) > capacity:
            memory_contexts.pop(0)

    data = {'actions': [], 'probs': [], 'rewards': [], 'values': [], 'readouts': []}
    for i in range(context_num):
        actions, probs, rewards, values, readouts = [], [], [], [], []
        for j in range(trials_per_condition):
            obs = torch.Tensor(env.reset(context_index=i)).to(device)
            done = False
            agent.set_encoding(False)
            state = agent.init_state(1)
            with analyze(agent):
                actions_trial, probs_trial, rewards_trial, values_trial = [], [], [], []
                while not done:
                    action_distribution, value, state = agent(obs, state)
                    action, log_prob_action = pick_action(action_distribution)
                    obs_, reward, done, info = env.step(action)
                    obs = torch.Tensor(obs_).to(device)

                    actions_trial.append(action)
                    probs_trial.append(log_prob_action)
                    rewards_trial.append(reward)
                    values_trial.append(value)
                actions.append(actions_trial)
                probs.append(probs_trial)
                rewards.append(rewards_trial)
                values.append(values_trial)
                readouts.append(agent.readout())
        data['actions'].append(actions)
        data['probs'].append(probs)
        data['rewards'].append(rewards)
        data['values'].append(values)
        data['readouts'].append(readouts)
    
    data['memory_contexts'] = memory_contexts
    
    return data
