import torch

from models.base_module import analyze
from .criterions.rl import pick_action


def record_model(agent, env, context_num=20, get_memory=False, device='cpu'):
    
    context_num = env.context_num if hasattr(env, "context_num") else context_num

    if hasattr(env, "regenerate_contexts"):
        env.regenerate_contexts()
    agent.set_retrieval(True)

    data = {'actions': [], 'probs': [], 'rewards': [], 'values': [], 'readouts': [],
    'memory_contexts': [], 'accuracy': 0.0}
    actions_total_num, actions_correct_num = 0, 0
    for i in range(context_num):
        # actions, probs, rewards, values, readouts, mem_contexts = [], [], [], [], [], []
        obs_, info = env.reset()
        agent.reset_memory()
        obs = torch.Tensor(obs_).to(device)
        done = False
        state = agent.init_state(1)
        with analyze(agent):
            actions_trial, probs_trial, rewards_trial, values_trial, actions_for_accuracy = [], [], [], [], []
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
                    state = agent.init_state(1, recall=True, prev_state=state)
                
                output, value, state, _ = agent(obs, state)
                if isinstance(output, tuple):
                    action_distribution = output[0]
                else:
                    action_distribution = output
                action, log_prob_action, action_max = pick_action(action_distribution)
                obs_, reward, done, info = env.step(action, batch_size=1)
                obs = torch.Tensor(obs_).to(device)

                actions_trial.append(int(action[0].detach().cpu()))
                actions_for_accuracy.append([int(action[0].detach().cpu())])
                probs_trial.append(log_prob_action)
                rewards_trial.append(reward)
                values_trial.append(value)
            readout = agent.readout()
            mem_seq = env.memory_sequence
            # actions.append(actions_trial)
            # probs.append(probs_trial)
            # rewards.append(rewards_trial)
            # values.append(values_trial)
            # readouts.append(agent.readout())
            # mem_contexts.append(env.memory_sequence)
            # print(actions_trial)
            correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions_for_accuracy)
            actions_total_num += correct_actions + wrong_actions + not_know_actions
            actions_correct_num += correct_actions
        data['actions'].append(actions_trial)
        data['probs'].append(probs_trial)
        data['rewards'].append(rewards_trial)
        data['values'].append(values_trial)
        data['readouts'].append(readout)
        data['memory_contexts'].append(mem_seq)
    print("test accuracy: {}".format(actions_correct_num/actions_total_num))
    data['accuracy'] = actions_correct_num/actions_total_num
    
    return data
