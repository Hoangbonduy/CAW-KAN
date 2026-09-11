import argparse
import os
import statistics
import torch
import torch.backends
from utils.print_args import print_args
import random
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_JDKAN import Exp_JDKAN  # Lazy import để tránh lỗi khi không dùng JDKAN
import numpy as np


def mean_std(values):
    if not values:
        raise ValueError('Cannot calculate statistics for an empty list')
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def format_mean_std(values):
    mean, std = mean_std(values)
    return f'{mean:.6f} +- {std:.6f}'


def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def echo_seed_status(stage, run_seed, ii):
    os.system(f'echo "[{stage}] itr={ii} seed={run_seed}"')


def parse_seed_list(seed_text):
    seeds = []
    for item in seed_text.split(','):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError('seed_list is empty. Example: 2021,2022,2023')
    return seeds


def build_setting(args, ii, run_seed):
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_seed{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        run_seed,
        ii,
    )

if __name__ == '__main__':
    set_global_seed(2021)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # JD-KAN
    parser.add_argument('--kan_order', type=int, default=3, help='Bậc của đa thức trong rKAN')
    parser.add_argument('--n_fourier_terms', type=int, default=8, help='Số lượng Fourier terms cho SmoothLayer')
    parser.add_argument('--rkan_order', type=int, default=3, help='Bậc của rKAN cho DiffusionLayer')
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat',
                        choices=['mexican_hat', 'morlet', 'dog', 'shannon'],
                        help='Loại wavelet cho AdaptiveWaveletKAN')
    parser.add_argument('--degree', type=int, default=3, help='Bậc n của đa thức Chebyshev. Tổng số sóng sinh ra = 2n + 1')
    parser.add_argument('--num_wavelets', type=int, default=8, help='Số lượng wavelet cho AdaptiveWaveletKAN')
    parser.add_argument('--grid_size', type=float, default=3.0, help='Grid size for wavelets (grid_min=-grid_size, grid_max=grid_size)')
    parser.add_argument('--kernel_size', type=int, default=7,
                        help='Conv1D kernel size for MS_JDKAN single context branch (default: 7)')
    parser.add_argument('--rank', type=int, default=8, help='Rank for CP Factorization in AdaptiveWaveletKAN')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for AdamW optimizer')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping max norm (0=disabled)')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start for OneCycleLR (used with lradj=TST)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2021, help="Randomization seed")

    # Run 3 seeds
    parser.add_argument('--run_three_seeds', action='store_true', default=False,
                        help='run default seeds 2021,2022,2023 and report mean+-std (default: on)')
    parser.add_argument('--no_run_three_seeds', action='store_false', dest='run_three_seeds',
                        help='disable automatic 3-seed mode')
    parser.add_argument('--seed_list', type=str, default='2021,2022,2023',
                        help='comma-separated seeds used when run_three_seeds is enabled')
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')
    
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    seed_runs = parse_seed_list(args.seed_list) if args.run_three_seeds else [args.seed]
    print('Seed schedule:', seed_runs)
    os.system(f'echo "[SEED_SCHEDULE] {seed_runs}"')


    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast 

    if args.is_training:
        seed_metrics = {ii: {'mse': [], 'mae': []} for ii in range(args.itr)}
        for ii in range(args.itr):
            for run_seed in seed_runs:
                args.seed = run_seed
                set_global_seed(run_seed)
                echo_seed_status('TRAIN', run_seed, ii)

                # setting record of experiments
                exp = Exp(args)  # set experiments
                setting = build_setting(args, ii, run_seed)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                echo_seed_status('TEST', run_seed, ii)
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                metric_path = os.path.join('./results', setting, 'metrics.npy')
                if os.path.exists(metric_path):
                    metrics = np.load(metric_path)
                    seed_metrics[ii]['mae'].append(float(metrics[0]))
                    seed_metrics[ii]['mse'].append(float(metrics[1]))

                if args.use_gpu:
                    if args.gpu_type == 'mps':
                        torch.backends.mps.empty_cache()
                    elif args.gpu_type == 'cuda':
                        torch.cuda.empty_cache()

        if args.run_three_seeds:
            summary_lines = []
            print('\n========== 3-SEED SUMMARY (mean +- std) ==========' )
            for ii in range(args.itr):
                mse_values = seed_metrics[ii]['mse']
                mae_values = seed_metrics[ii]['mae']
                if len(mse_values) != len(seed_runs) or len(mae_values) != len(seed_runs):
                    raise ValueError(
                        f'Missing seed results at itr={ii}: mse={len(mse_values)}, mae={len(mae_values)}, expected={len(seed_runs)}'
                    )

                line = (
                    f'itr={ii}, seeds={seed_runs}, '
                    f'mse={format_mean_std(mse_values)}, '
                    f'mae={format_mean_std(mae_values)}'
                )
                summary_lines.append(line)
                print(line)

            with open('result_long_term_forecast.txt', 'a') as f:
                f.write('========== 3-SEED SUMMARY (mean +- std) ==========' + '\n')
                for line in summary_lines:
                    f.write(line + '\n')
                f.write('\n')
    else:
        seed_metrics = {'mse': [], 'mae': []}
        ii = 0
        for run_seed in seed_runs:
            args.seed = run_seed
            set_global_seed(run_seed)
            echo_seed_status('TEST_ONLY', run_seed, ii)

            exp = Exp(args)  # set experiments
            setting = build_setting(args, ii, run_seed)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)

            metric_path = os.path.join('./results', setting, 'metrics.npy')
            if os.path.exists(metric_path):
                metrics = np.load(metric_path)
                seed_metrics['mae'].append(float(metrics[0]))
                seed_metrics['mse'].append(float(metrics[1]))

            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()

        if args.run_three_seeds:
            if len(seed_metrics['mse']) != len(seed_runs) or len(seed_metrics['mae']) != len(seed_runs):
                raise ValueError(
                    f'Missing seed results in test mode: mse={len(seed_metrics["mse"])}, '
                    f'mae={len(seed_metrics["mae"])}, expected={len(seed_runs)}'
                )
            print('\n========== 3-SEED SUMMARY (mean +- std) ==========' )
            print(
                f'seeds={seed_runs}, '
                f'mse={format_mean_std(seed_metrics["mse"])}, '
                f'mae={format_mean_std(seed_metrics["mae"])}'
            )
