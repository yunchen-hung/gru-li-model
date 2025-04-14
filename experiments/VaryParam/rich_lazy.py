import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from train.criterions.rl import pick_action
from models.utils import entropy



def analyze_parameter_change(model, model_checkpoints, save_path):
    """Analyze whether model is in rich or lazy regime by tracking NTK over training"""
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    param_norms = []
    
    # Get initial NTK and parameters
    model.load_state_dict(model_checkpoints[0])
    init_params = torch.cat([p.data.flatten() for p in model.parameters()])
    
    # Track changes during training
    for i, model_checkpoint in enumerate(model_checkpoints):
        model.load_state_dict(model_checkpoint)
        
        # # Compute current NTK and parameters
        curr_params = torch.cat([p.data.flatten() for p in model.parameters()])
        
        # # Compute relative changes
        param_diff = torch.norm(curr_params - init_params) / torch.norm(init_params)

        param_norms.append(param_diff.item())
    
    # Save numerical results
    np.save(save_path / 'param_changes.npy', np.array(param_norms))
    
    return param_norms



def generate_data(env, num_trials):
    """
    randomly generate the memory sequence index for a bunch of trials
    """
    data = []
    for _ in range(num_trials):
        env.reset()
        data.append(env.memory_sequence_indexs)
    return data


def compute_ntk(model, env, criterion, data):
    """Compute Neural Tangent Kernel"""

    # Get parameters
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    
    # Initialize NTK matrix
    ntk = None
    
    # Compute gradients for each data point
    grads = []
    for sequence in data:
        # Reset environment and model state
        obs_, info =env.reset(memory_sequence_index=sequence)
        obs = torch.Tensor(obs_).reshape(1, -1)
        done = False
        model.reset_memory()
        state = model.init_state(1)

        loss_masks, outputs, values, rewards, probs, entropys = [], [], [], [], [], []
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
            obs_, reward, _, _, info = env.step(action.cpu().detach().numpy().squeeze(axis=1))
            obs = torch.Tensor(obs_).reshape(1, -1)
            loss_masks.append(info["loss_mask"][0] and not done)
            outputs.append(output)
            values.append(value)
            rewards.append(reward)
            probs.append(log_prob_action)
            entropys.append(entropy(output))

            done = info["done"]

        loss_rl, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards[memory_num:], entropys, 
                                                                    loss_masks[memory_num:], print_info=False)
        
        loss_rl.backward()
        grad = torch.cat([p.grad.flatten() for p in params])
        grads.append(grad)
        
        # Zero gradients
        model.zero_grad()
    
    # Average NTK over samples
    grads = torch.stack(grads)
    ntk = grads.T @ grads
    
    return ntk


def analyze_ntk_change(model, model_checkpoints, env, criterion, save_path):
    """Analyze whether model is in rich or lazy regime by tracking NTK over training"""
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    ntk_norms = []
    data = generate_data(env, 100)
    
    # Get initial NTK and parameters
    init_ntk = compute_ntk(model, env, criterion, data)
    model.load_state_dict(model_checkpoints[0])
    
    # Track changes during training
    for i, model_checkpoint in enumerate(model_checkpoints):
        # x_batch, _ = next(iter(dataloader))
        model.load_state_dict(model_checkpoint)
        
        # # Compute current NTK and parameters
        curr_ntk = compute_ntk(model, env, criterion, data) 
        
        # # Compute relative changes
        ntk_diff = torch.norm(curr_ntk - init_ntk) / torch.norm(init_ntk)
        
        ntk_norms.append(ntk_diff.item())
    
    # Save numerical results
    np.save(save_path / 'ntk_changes.npy', np.array(ntk_norms))

    return ntk_norms


def run(data_all, model_all, env, paths, exp_name, checkpoints=None, criterion=None):
    plt.rcParams['font.size'] = 16

    env = env[0]

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        # fig_path = paths["fig"]/run_name
        run_num = run_name.split("-")[-1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)
        print()
        print(run_name)

        model = model_all[run_name]
        checkpoints_model = checkpoints[run_name]
        checkpoint_session_nums, checkpoint_epoch_nums, checkpoints = checkpoints_model[0], checkpoints_model[1], checkpoints_model[2]
        if checkpoints_model is not None:
            checkpoint_labels = ['{}_{}'.format(session_num, epoch_num) for session_num, epoch_num in zip(checkpoint_session_nums, checkpoint_epoch_nums)]

            # analyze parameter change
            param_norms = analyze_parameter_change(model, checkpoints_model, fig_path)
            plt.figure(figsize=(4, 3.3), dpi=180)
            plt.plot(checkpoint_labels, param_norms)
            plt.xticks(rotation=45)
            plt.xlabel('Training Steps')
            plt.ylabel('Relative Parameter Change')
            plt.tight_layout()
            plt.savefig(fig_path / 'param_change.png')
            plt.close()

            # analyze ntk change
            ntk_norms = analyze_ntk_change(model, checkpoints_model, env, criterion, fig_path)
            plt.figure(figsize=(4, 3.3), dpi=180)
            plt.plot(checkpoint_labels, ntk_norms)
            plt.xticks(rotation=45)
            plt.xlabel('Training Steps')
            plt.ylabel('Relative NTK Change')
            plt.tight_layout()
            plt.savefig(fig_path / 'ntk_change.png')
            plt.close()
