import torch
import torch.fft
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from models import AHuber
from data_provider.data_factory import data_provider
from scipy.linalg import subspace_angles

# ==========================================
# 1. Parameter Definition (Consistent with vis_analysis.py)
# ==========================================
parser = argparse.ArgumentParser(description='AHuber Spectrum Visualization')

# Basic Configuration
parser.add_argument('--model', type=str, default='AHuber', help='model name')
parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# Core model parameters (Must be exactly the same as during training!)
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=321, help='encoder input size')
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
parser.add_argument('--model_id', type=str, default='electricity_96_96', help='model id')
parser.add_argument('--data', type=str, default='electricity', help='dataset type')
parser.add_argument('--features', type=str, default='M', help='forecasting task')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
parser.add_argument('--revin', type=int, default=1, help='whether to use RevIN (1:Yes, 0:No)')
parser.add_argument('--affine', type=int, default=1, help='whether to use affine in RevIN (1:Yes, 0:No)')
parser.add_argument('--subtract_last', type=int, default=0, help='RevIN mode (0: subtract mean, 1: subtract last)')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
args = parser.parse_args()
args.output_hidden_states = True # Force enable

# ==========================================
# 2. Model Loading
# ==========================================
def load_model(args):
    # Manually construct model_name_string to find the checkpoint (manually fill in the path if not found)
    setting = '{}_{}_{}_ft{}_sl{}_pd{}_dm{}_nh{}_el{}_df{}_lr{}_dp{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.pred_len, 
        args.d_model, args.n_heads, args.e_layers, args.d_ff, 
        args.learning_rate, args.dropout
    )
    # If path is incorrect, manually hardcode it here:
    # best_model_path = r"your absolute checkpoint path"
    best_model_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    
    print(f">>> Attempting to load: {best_model_path}")

    model = AHuber.Model(args)
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location='cpu')
        # Compatibility for keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'): new_k = k.replace('model.', 'backbone.')
            elif not k.startswith('backbone.'): new_k = 'backbone.' + k
            else: new_k = k
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(">>> Checkpoint loaded successfully!")
    else:
        print("!!! Warning: Checkpoint not found. Using RANDOM weights (Graph shape is correct, but spectrum meaningless).")
    
    model.eval()
    if args.use_gpu: model.cuda()
    return model

def analyze_subspace_orthogonality(model, test_loader):
    print(">>> Running MULTI-LAYER Semantic Subspace Analysis...")
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    
    # Store data for all layers and all batches
    # Structure: layers_stats[layer_idx]['angles'] = list of batch_means
    layers_stats = {} 
    
    k = 20
    max_batches = 10
    num_layers = args.e_layers # Usually 3 or 4
    
    with torch.no_grad():
        for i, (batch_x, _, _, _) in enumerate(test_loader):
            if i >= max_batches: break
            batch_x = batch_x.float().to(device)
            _, (hidden_states, repair, _) = model(batch_x)
            
            # --- Loop through each layer ---
            for l_idx in range(num_layers):
                if l_idx not in layers_stats:
                    layers_stats[l_idx] = {'angles': [], 'sv_upd': [], 'sv_res': []}
                
                h = hidden_states[l_idx]
                r = repair[l_idx]
                
                Residual_Term = (h - r).reshape(-1, args.seq_len).cpu().numpy()
                Update_Term = r.reshape(-1, args.seq_len).cpu().numpy()
                
                U_upd, S_upd, _ = np.linalg.svd(Update_Term.T, full_matrices=False)
                U_res, S_res, _ = np.linalg.svd(Residual_Term.T, full_matrices=False)
                
                Basis_upd = U_upd[:, :k]
                Basis_res = U_res[:, :k]
                
                S_upd_norm = S_upd[:k] / np.sum(S_upd)
                S_res_norm = S_res[:k] / np.sum(S_res)
                
                angles_rad = subspace_angles(Basis_upd, Basis_res)
                angles_deg = np.rad2deg(angles_rad)
                
                layers_stats[l_idx]['angles'].append(angles_deg)
                layers_stats[l_idx]['sv_upd'].append(S_upd_norm)
                layers_stats[l_idx]['sv_res'].append(S_res_norm)
            
            if (i+1) % 5 == 0: print(f"Processed {i+1} batches...")

    # === Data Aggregation ===
    # Merge data from all layers, prepare for plotting average
    # global_angles: [num_layers * num_batches, k]
    global_angles = []
    global_sv_upd = []
    global_sv_res = []
    
    for l_idx in range(num_layers):
        # Data for all batches in this layer
        l_ang = np.array(layers_stats[l_idx]['angles']) # [batches, k]
        l_upd = np.array(layers_stats[l_idx]['sv_upd'])
        l_res = np.array(layers_stats[l_idx]['sv_res'])
        
        # Add the mean of this layer to the global list (or add all samples, depending on what you want to average)
        # Suggestion: Take out the mean of all layers, calculate inter-layer mean and std
        global_angles.append(np.mean(l_ang, axis=0))
        global_sv_upd.append(np.mean(l_upd, axis=0))
        global_sv_res.append(np.mean(l_res, axis=0))

    # Now global_xxx is [num_layers, k]
    global_angles = np.array(global_angles)
    global_sv_upd = np.array(global_sv_upd)
    global_sv_res = np.array(global_sv_res)
    
    # Calculate mean and std (cross-layer)
    mean_ang = np.mean(global_angles, axis=0)
    std_ang = np.std(global_angles, axis=0)
    
    # Added: Calculate the overall average of all Rank angles for display
    overall_avg_deg = np.mean(mean_ang)

    mean_upd = np.mean(global_sv_upd, axis=0)
    std_upd = np.std(global_sv_upd, axis=0)
    
    mean_res = np.mean(global_sv_res, axis=0)
    std_res = np.std(global_sv_res, axis=0)

    # === Plotting ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(1, k+1)
    
    # Figure 1: Cross-layer Average Orthogonality
    ax1.plot(x, mean_ang, 'o-', color='#1f77b4', linewidth=3, label='Avg Angle (All Layers)')
    ax1.fill_between(x, mean_ang - std_ang, mean_ang + std_ang, color='#1f77b4', alpha=0.2, label='Layer Variance')
    ax1.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 100)
    # Modified: Display average angle value in Title
    ax1.set_title(f'Subspace Orthogonality (Mean: {overall_avg_deg:.2f}°)', fontweight='bold', fontsize=18)
    ax1.set_xlabel('Rank Index', fontsize=18)
    ax1.set_ylabel('Degrees', fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14) # Adjust tick font size
    
    # Figure 2: Cross-layer Average Energy Spectrum
    # Hub Update
    ax2.plot(x, mean_upd, 's-', color='darkred', linewidth=2, label='Hub State (Avg)')
    ax2.fill_between(x, mean_upd - std_upd, mean_upd + std_upd, color='darkred', alpha=0.15, label='Hub Variance')
    
    # Residual
    ax2.plot(x, mean_res, '^-', color='darkgreen', linewidth=2, label='Residual Input (Avg)')
    ax2.fill_between(x, mean_res - std_res, mean_res + std_res, color='darkgreen', alpha=0.15, label='Residual Variance')
    
    ax2.set_title('Singular Value Spectrum', fontweight='bold', fontsize=18)
    ax2.set_xlabel('Rank Index', fontsize=18)
    ax2.set_ylabel('Energy Ratio', fontsize=18)
    ax2.legend(fontsize=14) 
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=14) # Adjust tick font size
    
    plt.tight_layout()
    plt.show()

    # plt.savefig('orthogonality_analysis.pdf')
    
if __name__ == "__main__":
    _, test_loader = data_provider(args, flag='test')
    model = load_model(args)
    analyze_subspace_orthogonality(model, test_loader)
