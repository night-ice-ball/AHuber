import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AHuber')

    # ===== Basic Config =====
    parser.add_argument('--is_training', type=int, default=1, help='1: training, 0: testing')
    parser.add_argument('--model_id', type=str, default='weather_96_96', help='model ID, used for saving')
    parser.add_argument('--model', type=str, default='AHuber', 
                    help='model name, options: [AHuber, iTransformer, PatchTST, DLinear, ...]')
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # ===== Data Loader Config =====
    parser.add_argument('--data', type=str, default='weather', help='dataset name (e.g., ETTm1, ETTh1, ECL)')
    parser.add_argument('--root_path', type=str, default='./data/weather/', help='root path of the data file')
    # --- data_path and enc_in will be set automatically ---
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file name')
    parser.add_argument('--enc_in', type=int, default=7, help='number of input variables')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding: s, t, h, d, b, w, m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    # ===== Forecasting Task Config =====
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')

    # ===== AHuber Model Core Params =====
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=16, help='patch stride')
    parser.add_argument('--e_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # --- RevIN Params ---
    parser.add_argument('--revin', type=int, default=1, help='whether to use RevIN (1:Yes, 0:No)')
    parser.add_argument('--affine', type=int, default=1, help='whether to use affine in RevIN (1:Yes, 0:No)')
    parser.add_argument('--subtract_last', type=int, default=0, help='RevIN mode (0: subtract mean, 1: subtract last)')

    # ===== Optimizer & Training Config =====
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=16, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')

    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type2', help='learning rate adjustment strategy')
    parser.add_argument('--pct_start', type=float, default=0.4, help='learning rate adjustment start percentage')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of mult gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    
    args = parser.parse_args()

    # === Auto-configure params based on dataset ===
    data_info = {
        'ETTh1': {'data_path': 'ETTh1.csv', 'enc_in': 7, 'freq': 'h'},
        'ETTh2': {'data_path': 'ETTh2.csv', 'enc_in': 7, 'freq': 'h'},
        'ETTm1': {'data_path': 'ETTm1.csv', 'enc_in': 7, 'freq': 't'},
        'ETTm2': {'data_path': 'ETTm2.csv', 'enc_in': 7, 'freq': 't'},
        'electricity': {'data_path': 'electricity.csv', 'enc_in': 321, 'freq': 'h'},
        'weather': {'data_path': 'weather.csv', 'enc_in': 21, 'freq': 'h'},
        'traffic': {'data_path': 'traffic.csv', 'enc_in': 862, 'freq': 'h'},
        'exchange': {'data_path': 'exchange_rate.csv', 'enc_in': 8, 'freq': 'd'},
        'solar': {'data_path': 'solar_energy.csv', 'enc_in': 7, 'freq': 'h'},
        # Add other dataset info here
    }
    if args.data in data_info:
        info = data_info[args.data]
        args.data_path = info['data_path']
        args.enc_in = info['enc_in']
        args.freq = info['freq'] # Auto set freq
    else:
        # If custom dataset, manually provide data_path and enc_in
        if args.data == 'custom' and (not hasattr(args, 'data_path') or not hasattr(args, 'enc_in')):
             raise ValueError("For custom dataset, --data_path and --enc_in must be provided.")

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    # === Optimization: Define setting string once ===
    setting = '{}_{}_{}_ft{}_sl{}_pd{}_pl{}_sd{}_dm{}_nh{}_el{}_df{}_lr{}_dp{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.patch_len,
        args.stride,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        args.learning_rate,
        args.dropout,
    )

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
