import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from train.criterions.rl import pick_action
from models.utils import entropy



def analyze_parameter_change(model, model_checkpoints, save_path, layer_names=["encoder", "hidden", "decoder"]):
    """Analyze whether model is in rich or lazy regime by tracking NTK over training"""
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    param_norms = []
    param_norms_by_layer = []
    
    # Get initial NTK and parameters
    model.load_state_dict(model_checkpoints[0])

    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Parameter shape: {param.shape}")

    init_params = torch.cat([p.data.flatten() for p in model.parameters()])
    init_params_by_layer = []
    for layer in layer_names:
        init_params_by_layer.append(torch.cat([p.data.flatten() for name, p in model.named_parameters() if layer in name]))
    
    # Track changes during training
    for i, model_checkpoint in enumerate(model_checkpoints):
        model.load_state_dict(model_checkpoint)
        
        # # Compute current NTK and parameters
        curr_params = torch.cat([p.data.flatten() for p in model.parameters()])
        curr_params_by_layer = []
        for layer in layer_names:
            curr_params_by_layer.append(torch.cat([p.data.flatten() for name, p in model.named_parameters() if layer in name]))
        
        # # Compute relative changes
        param_diff = torch.norm(curr_params - init_params) / torch.norm(init_params)
        param_diff_by_layer = []
        for j, layer in enumerate(layer_names):
            param_diff_by_layer.append(torch.norm(curr_params_by_layer[j] - init_params_by_layer[j]) / torch.norm(init_params_by_layer[j]))

        # # cumulative
        init_params = curr_params
        init_params_by_layer = curr_params_by_layer

        param_norms.append(param_diff.item())
        param_norms_by_layer.append([param_diff_by_layer[j].item() for j in range(len(layer_names))])

    param_norms = np.array(param_norms)
    param_norms_by_layer = np.array(param_norms_by_layer)

    param_norms = np.cumsum(param_norms)
    param_norms_by_layer = np.cumsum(param_norms_by_layer, axis=0)

    # Save numerical results
    np.save(save_path / 'param_changes.npy', param_norms)
    np.save(save_path / 'param_changes_by_layer.npy', param_norms_by_layer)
    
    return param_norms, param_norms_by_layer



def generate_data(env, num_trials):
    """
    randomly generate the memory sequence index for a bunch of trials
    """
    data = []
    for _ in range(num_trials):
        env.reset()
        data.append(env.unwrapped.memory_sequence_index)
    return data


def compute_ntk(model, env, criterion, data, layer_names=["encoder", "hidden", "decoder"]):
    """Compute Neural Tangent Kernel"""

    # Get parameters
    params = []
    # print(model)
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    # for p in params:
    #     print(p.shape)

    params_by_layer = []
    for layer in layer_names:
        params_by_layer.append([p for name, p in model.named_parameters() if layer in name])
    
    # Initialize NTK matrix
    ntk = None
    
    # Compute gradients for each data point
    grads = []
    grads_by_layer = []
    for sequence in data:
        # Reset environment and model state
        obs_, info =env.reset(memory_sequence_index=sequence)
        obs = torch.Tensor(obs_).reshape(1, -1)
        done = False
        model.reset_memory()
        state = model.init_state(1)

        loss_masks, rewards = [], []
        probs, values, entropys = defaultdict(list), defaultdict(list), defaultdict(list)
        memory_num = 0
        while not done:
            # set up the phase of the model 
            if info["phase"] == "encoding":
                model.set_encoding(True)
                model.set_retrieval(False)
                memory_num += 1
            elif info["phase"] == "recall":
                model.set_encoding(False)
                model.set_retrieval(True)
            # reset state between phases
            if "reset_state" in info and info["reset_state"]:
                state = model.init_state(1, recall=True, prev_state=state)
            
            output, value, state, _ = model(obs, state)
            action_distribution = output
            action, log_prob_action, action_max = pick_action(action_distribution)
            obs_, reward, _, _, info = env.step(action_max.cpu().detach().numpy().squeeze(axis=1))
            obs = torch.Tensor(obs_).reshape(1, -1)
            loss_masks.append([info["loss_mask"] and not done])
            values[0].append(value[0])
            rewards.append([reward])
            probs[0].append(log_prob_action[0])
            entropys[0].append(entropy(output[0]))

            done = info["done"]

        loss_masks = np.array(loss_masks)
        probs[0] = probs[0][memory_num:]
        values[0] = values[0][memory_num:]
        rewards = rewards[memory_num:]
        entropys[0] = entropys[0][memory_num:]
        loss_masks = loss_masks[memory_num:]
        loss_rl, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards, entropys, 
                                                                    loss_masks, print_info=False)
        
        loss_rl.backward()
        # for p in params:
        #     print(p.shape, p.grad)
        grad = torch.cat([p.grad.flatten().to(torch.float32) for p in params if p.grad is not None])
        grads.append(grad)

        grad_by_layer = []
        for i, layer in enumerate(layer_names):
            grad_by_layer.append(torch.cat([p.grad.flatten().to(torch.float32) for p in params_by_layer[i] if p.grad is not None]))
        grads_by_layer.append(grad_by_layer)
        
        # Zero gradients
        model.zero_grad()


    # Average NTK over samples
    grads = torch.stack(grads)
    ntk = grads @ grads.T

    grads_by_layer2 = []
    for i, layer in enumerate(layer_names):
        grads_by_layer2.append(torch.stack([grad_by_layer[i] for grad_by_layer in grads_by_layer]))

    ntk_by_layer = []
    for i, layer in enumerate(layer_names):
        ntk_by_layer.append(grads_by_layer2[i] @ grads_by_layer2[i].T)

    # del grads
    
    return ntk, ntk_by_layer


def analyze_ntk_change(model, model_checkpoints, env, criterion, save_path, layer_names=["encoder", "hidden", "decoder"]):
    """Analyze whether model is in rich or lazy regime by tracking NTK over training"""
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    ntk_norms = []
    ntk_norms_by_layer = []
    data = generate_data(env, 1000)
    
    model.load_state_dict(model_checkpoints[0])
    # Get initial NTK and parameters
    init_ntk, init_ntk_by_layer = compute_ntk(model, env, criterion, data, layer_names)
    print(torch.norm(init_ntk))
    
    # Track changes during training
    for i, model_checkpoint in enumerate(model_checkpoints):
        print("checkpoint", i)

        # x_batch, _ = next(iter(dataloader))
        model.load_state_dict(model_checkpoint)
        
        # # Compute current NTK and parameters
        curr_ntk, curr_ntk_by_layer = compute_ntk(model, env, criterion, data, layer_names) 
        
        # # Compute relative changes
        ntk_diff = torch.norm(curr_ntk - init_ntk) / torch.norm(init_ntk)
        ntk_diff_by_layer = []
        for i, layer in enumerate(layer_names):
            ntk_diff_by_layer.append(torch.norm(curr_ntk_by_layer[i] - init_ntk_by_layer[i]) / torch.norm(init_ntk_by_layer[i]))

        ntk_norms.append(ntk_diff.item())
        ntk_norms_by_layer.append([ntk_diff_by_layer[i].item() for i in range(len(layer_names))])

    ntk_norms = np.array(ntk_norms)
    ntk_norms_by_layer = np.array(ntk_norms_by_layer)

    # Save numerical results
    np.save(save_path / 'ntk_changes.npy', ntk_norms)
    np.save(save_path / 'ntk_changes_by_layer.npy', ntk_norms_by_layer)

    return ntk_norms, ntk_norms_by_layer


def run(data_all, model_all, env, paths, exp_name, checkpoints=None, criterion=None):
    plt.rcParams['font.size'] = 16

    env = env[0]

    layer_names = ["encoder", "hidden", "decoder"]

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        # fig_path = paths["fig"]/run_name
        run_num = run_name.split("-")[-1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)
        print()
        print(run_name)

        model = model_all[run_name]
        checkpoint = checkpoints[run_name]
        checkpoint_session_nums, checkpoint_epoch_nums, checkpoints_model = checkpoint[0], checkpoint[1], checkpoint[2]

        checkpoint_labels = ['{}_{}'.format(epoch_num, session_num) for session_num, epoch_num in zip(checkpoint_session_nums, checkpoint_epoch_nums)]


        # analyze parameter change
        param_norms, param_norms_by_layer = analyze_parameter_change(model, checkpoints_model, fig_path, layer_names)
        # print(param_norms.shape, param_norms_by_layer.shape)
        plt.figure(figsize=(5, 4), dpi=180)
        plt.plot(checkpoint_labels, param_norms)
        plt.xticks(rotation=45, fontsize=9)
        plt.xlabel('Training Steps')
        plt.ylabel('Relative Parameter\nChange')
        plt.tight_layout()
        plt.savefig(fig_path / 'param_change.png')
        plt.close()

        plt.figure(figsize=(5, 4), dpi=180)
        for i, layer in enumerate(layer_names):
            plt.plot(checkpoint_labels, param_norms_by_layer[:, i], label=layer)
        plt.xticks(rotation=45, fontsize=9)
        plt.xlabel('Training Steps')
        plt.ylabel('Relative Parameter\nChange')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_path / 'param_change_by_layer.png')
        plt.close()


        # analyze ntk change
        ntk_norms, ntk_norms_by_layer = analyze_ntk_change(model, checkpoints_model, env, criterion, fig_path, layer_names)

        # delete results with index 2
        ntk_norms = ntk_norms[2:]
        ntk_norms_by_layer = ntk_norms_by_layer[2:]
        checkpoint_labels = checkpoint_labels[2:]

        plt.figure(figsize=(5, 4), dpi=180)
        plt.plot(checkpoint_labels, ntk_norms)
        plt.xticks(rotation=45, fontsize=9)
        plt.xlabel('Training Steps')
        plt.ylabel('Relative NTK Change')
        plt.tight_layout()
        plt.savefig(fig_path / 'ntk_change.png')
        plt.close()


        plt.figure(figsize=(5, 4), dpi=180)
        for i, layer in enumerate(layer_names):
            plt.plot(checkpoint_labels, ntk_norms_by_layer[:, i], label=layer)
        plt.xticks(rotation=45, fontsize=9)
        plt.xlabel('Training Steps')
        plt.ylabel('Relative NTK Change')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_path / 'ntk_change_by_layer.png')
        plt.close()
