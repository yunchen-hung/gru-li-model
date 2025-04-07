import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_data(env, num_trials):
    """
    randomly generate the memory sequence index for a bunch of trials
    """
    data = []
    for _ in range(num_trials):
        env.reset()
        data.append(env.memory_sequence_indexs)
    return data


def compute_ntk(model, env, data):
    """Compute Neural Tangent Kernel"""
    # Generate random data
    data = generate_data(env, 10)  # Use 10 trials for NTK computation
    
    # Get parameters
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    
    # Initialize NTK matrix
    ntk = None
    
    # Compute gradients for each data point
    for sequence in data:
        # Reset environment and model state
        env.reset()
        model.reset_states()
        
        # Run model forward and get loss
        total_loss = 0
        for t in range(len(sequence)):
            # Get observation and run model
            obs = env.get_observation()
            output = model(obs)
            
            # Compute loss (using dummy target)
            target = torch.zeros_like(output)
            loss = ((output - target)**2).sum()
            total_loss += loss
            
            # Step environment
            env.step(0)  # Dummy action
            
        # Compute gradients
        total_loss.backward()
        
        # Get flattened gradient
        grad = torch.cat([p.grad.flatten() for p in params])
        
        # Update NTK
        if ntk is None:
            ntk = grad.unsqueeze(0).T @ grad.unsqueeze(0)
        else:
            ntk += grad.unsqueeze(0).T @ grad.unsqueeze(0)
        
        # Zero gradients
        model.zero_grad()
    
    # Average NTK over samples
    ntk = ntk / len(data)
    
    return ntk


def analyze_training_regime(model, model_checkpoints, env, save_path):
    """Analyze whether model is in rich or lazy regime by tracking NTK over training"""
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    ntk_norms = []
    param_norms = []
    
    # Get initial NTK and parameters
    # x_batch, _ = next(iter(dataloader))
    # init_ntk = compute_ntk(model, x_batch)
    model.load_state_dict(model_checkpoints[0])
    init_params = torch.cat([p.data.flatten() for p in model.parameters()])
    
    # Track changes during training
    for i, model_checkpoint in enumerate(model_checkpoints):
        # x_batch, _ = next(iter(dataloader))
        model.load_state_dict(model_checkpoint)
        
        # # Compute current NTK and parameters
        # curr_ntk = compute_ntk(model, x_batch) 
        curr_params = torch.cat([p.data.flatten() for p in model.parameters()])
        
        # # Compute relative changes
        # ntk_diff = torch.norm(curr_ntk - init_ntk) / torch.norm(init_ntk)
        param_diff = torch.norm(curr_params - init_params) / torch.norm(init_params)
        
        # ntk_norms.append(ntk_diff.item())
        param_norms.append(param_diff.item())
        
    # Plot results
    plt.figure(figsize=(4, 3.3), dpi=180)
    # plt.plot(ntk_norms, label='NTK Change')
    plt.plot(param_norms, label='Parameter Change') 
    plt.xlabel('Training Steps')
    plt.ylabel('Relative Change')
    plt.legend()
    # plt.title('NTK vs Parameter Changes During Training')
    plt.tight_layout()
    plt.savefig(save_path / 'param_change.png')
    plt.close()
    
    # Save numerical results
    # np.save(save_path / 'ntk_changes.npy', np.array(ntk_norms))
    np.save(save_path / 'param_changes.npy', np.array(param_norms))
    
    # Determine regime
    # ntk_final = np.mean(ntk_norms[-10:])  # Average of last 10 steps
    # if ntk_final < 0.1:  # Threshold can be adjusted
    #     print("Model appears to be in lazy regime")
    #     regime = "lazy"
    # else:
    #     print("Model appears to be in rich regime") 
    #     regime = "rich"
        
    # return regime, ntk_norms, param_norms
    return param_norms


def run(data_all, model_all, env, paths, exp_name, checkpoints=None):
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
        if checkpoints_model is not None:
            analyze_training_regime(model, checkpoints_model, env, fig_path)
