import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models import AHuber 
from data_provider.data_factory import data_provider 
# ==========================================
# 1. Parameter Definition (Copied from your run.py or main.py)
# ==========================================
parser = argparse.ArgumentParser(description='AHuber Visualization')

# Basic Configuration
parser.add_argument('--model', type=str, default='AHuber', help='model name')
parser.add_argument('--root_path', type=str, default='./data/traffic/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# Core model parameters (Must be exactly the same as during training!)
parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=862, help='encoder input size')
parser.add_argument('--d_model', type=int, default=192, help='dimension of model')
parser.add_argument('--batch_size', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=12, help='num of heads')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=384, help='dimension of fcn')
parser.add_argument('--d_hub', type=int, default=1, help='dimension of hub')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding: s, t, h, d, b, w, m') 
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=16, help='patch stride')

# Extra parameters, used to match the logic of generating the setting string in exp_main.py
parser.add_argument('--model_id', type=str, default='traffic_720_96', help='model id')
parser.add_argument('--data', type=str, default='traffic', help='dataset type')
parser.add_argument('--features', type=str, default='M', help='forecasting task')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
parser.add_argument('--revin', type=int, default=1, help='whether to use RevIN (1:Yes, 0:No)')
parser.add_argument('--affine', type=int, default=1, help='whether to use affine in RevIN (1:Yes, 0:No)')
parser.add_argument('--subtract_last', type=int, default=0, help='RevIN mode (0: subtract mean, 1: subtract last)')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--output_hidden_states', type=bool, default=True, help='attention factor')

def load_model_from_args(args):
    """
    Reconstruct Setting string based on arguments, find path, and load weights
    """
    # 1. Force enable output_hidden_states for visualization
    args.output_hidden_states = True 
    
    # 2. Construct Setting string (Logic must be consistent with train function in Exp_Main)
    # setting format is usually: {model_id}_{model}_{data}_{features}_sl{}_pl{}_...
    setting = '{}_{}_{}_ft{}_sl{}_pd{}_dm{}_nh{}_el{}_df{}_lr{}_dp{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        args.learning_rate,
        args.dropout,
    )
    
    # 3. Find path
    best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')

    print(f">>> Loading model from: {best_model_path}")
    if not os.path.exists(best_model_path):
        print("!! Checkpoint not found. Initializing random model for testing/demo !!")

    # 4. Instantiate model
    model = AHuber.Model(args)
    
    # 5. Load weights
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=torch.device('cpu')) # Load to CPU by default to prevent VRAM issues
        model.load_state_dict(state_dict)
        print(">>> Model weights loaded successfully!")
        new_state_dict = {}
        for key, value in state_dict.items():

            if key.startswith("model."):
                new_key = key.replace("model.", "backbone.", 1)

            elif not key.startswith("backbone."):
                new_key = "backbone." + key
            else:
                new_key = key
                
            new_state_dict[new_key] = value
            
        # Load with processed new dictionary
        model.load_state_dict(new_state_dict)
        print(">>> Model weights loaded successfully (with key mapping)!")
    else:
        print("!! Checkpoint not found !!")
    model.eval()
    return model

# ==========================================
# 2. Parse arguments only when running as a script
# ==========================================
if __name__ == "__main__":
    # Simulate command line arguments (You can fill in real arguments used during training here)
    # Or run directly in terminal: python vis_analysis.py --model AHuber --d_model 512 ...
    args = parser.parse_args()
    

    model = load_model_from_args(args)

    # ==========================================
    # 3. Various visualization codes
    # ==========================================

    print(f">>> Generatng dummy input with shape: [1, {args.seq_len}, {args.enc_in}]")
     # ==========================================
    # 2. Get real data from data loader
    # ==========================================
    # Use the project's data_provider, it automatically handles scaler, timeenc, batching etc.
    print(f">>> Loading Test Data via data_provider...")
    # Note: args needs to contain correct info like root_path, data_path, seq_len
    # Default flag='test' so we are visualizing on the test set, logical
    test_set, test_loader = data_provider(args, flag='test')
    
    print(f"Test Set Size: {len(test_set)}")
    
    # Get one batch of data
    # enumerate returns (batch_x, batch_y, batch_x_mark, batch_y_mark)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        # We only need the first batch for plotting
        input_tensor = batch_x.float() # [Batch, Seq_Len, Vars]
        break
    
    print(f"Input Tensor Shape: {input_tensor.shape}")
    
    # If model loading fails, code runs with random stats, check path
    with torch.no_grad():
        # [Modified] Robust unpacking logic
        outputs = model(input_tensor)
        
        if isinstance(outputs, tuple):
            # Normal case: returned (pred, auxiliary_info)
            if len(outputs) == 2:
                pred, ret = outputs
                # Further unpack ret
                if isinstance(ret, tuple) and len(ret) == 3:
                     hidden_states, repair_terms, attn_weights = ret
                else:
                    print(f"Error: Aux return structure unexpected. Got type {type(ret)}")
                    exit()
            else:
                 print(f"Error: Expected 2 outputs, got {len(outputs)}")
                 exit()
        else:
            # Abnormal case: returned only pred
            print("Error: Model returned only prediction! 'output_hidden_states' flag might not be working.")
            print("Check models/AHuber.py to see if it utilizes configs.output_hidden_states correctly.")
            exit()

    sample_idx = 0
    
    # Prepare Input Sequence [Time, Vars]
    # input_tensor is [Batch, Seq, Vars]
    input_seq = input_tensor[sample_idx].numpy() # [Seq, Vars]
    
    # Let's print shape to confirm
    print(f"Hidden state L0 shape: {hidden_states[0].shape}")
    
    delta_plots = []
    output_plots = []
    
    for l_idx in range(len(hidden_states)):
        # h: [Batch, N, S] or [Batch, S, N]
        h = hidden_states[l_idx][sample_idx].detach().cpu().numpy()
        r = repair_terms[l_idx][sample_idx].detach().cpu().numpy()
        
        # Unify to [Seq, Vars]
        if h.shape[0] == args.enc_in: h = h.T
        if r.shape[0] == args.enc_in: r = r.T
            
        output_plots.append(h)
        delta_plots.append(r)
        
    # --- Plot 1: Rank-Aware Decoupling ---
    print("\n>>> Plotting Rank-Aware Decoupling (Bypass vs Residual)...")
    var_idx = 120
    # Define Zoom area for details
    zoom_start = 75
    zoom_end = 150  
    sl = slice(zoom_start, zoom_end)
    
    # Color definition
    color_bypass = '#1f77b4'  # Solid blue
    color_hub = '#d62728'     # Aggressive red

    fig, axes = plt.subplots(args.e_layers, 1, figsize=(10, 2.5*args.e_layers), sharex=True)
    if args.e_layers == 1: axes = [axes] # Compatible with single layer
    
    curr_input = input_seq[:, var_idx] # Initial input (Layer 1 Bypass)
    
    for i in range(args.e_layers):
        ax = axes[i]
        
        # Data Preparation
        bypass_data = curr_input[sl]   # Low-Rank Skeleton
        hub_data = delta_plots[i][:, var_idx][sl] # High-Rank Refinement
        
        # === Core: Dual-axis plotting ===
        ax2 = ax.twinx()  # Create second Y axis sharing X
        
        # 1. Left axis plots Bypass (Skeleton)
        line1 = ax.plot(range(zoom_start, zoom_end), bypass_data, 
                        color=color_bypass, lw=2.5, alpha=0.6, label='Bypass (Signal Skeleton)')
        
        # 2. Right axis plots Hub Update (Refinement)
        line2 = ax2.plot(range(zoom_start, zoom_end), hub_data, 
                         color=color_hub, lw=1.5, ls='-', alpha=0.9, label='Hub ($\Delta X$)')
        
        # 3. Set grid and labels
        ax.set_title(f'Layer {i+1}', fontweight='bold', fontsize=14)
        ax.set_ylabel('Low-Rank Skeleton', color=color_bypass, fontweight='bold')
        ax2.set_ylabel('High-Rank Refinement', color=color_hub, fontweight='bold')
        
        # Beautify tick colors
        ax.tick_params(axis='y', labelcolor=color_bypass)
        ax2.tick_params(axis='y', labelcolor=color_hub)
        
        # 4. Merge legends
        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', fontsize=9)
        
        ax.grid(True, alpha=0.3)
        
        # Prepare for next layer: Input becomes Output of this layer
        curr_input = output_plots[i][:, var_idx]

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('vis_rank_decoupling.pdf')
    print("Saved vis_rank_decoupling.pdf")
