import gc
import os
import shutil
import time
import argparse
import warnings

import torch

os.environ.setdefault('OMP_NUM_THREADS', '1')

from train import train

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='UDDA parameter experiments optimized without changing training logic')
parser.add_argument('--dataset', default='MNIST-USPS')
parser.add_argument('--batch_size', default=4096, type=int)
parser.add_argument("--temperature_f", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--main_epochs", default=3, type=int)
parser.add_argument("--feature_dim", default=20, type=int)
parser.add_argument("--num_users", default=24, type=int)
parser.add_argument("--Dirichlet_alpha", default=9999, type=float)
parser.add_argument("--interval_epoch", default=32, type=int)
parser.add_argument("--participate", default=1.0, type=float)
parser.add_argument('--lay_nums', nargs='*', type=int, default=[6, 6, 6])
parser.add_argument('--l1_factor', default=0.0, type=float)
parser.add_argument('--l2_factor', default=0.01, type=float)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--repeated_experiment_epoch', default=5, type=int)

parser.add_argument('--eval_interval', default=10, type=int,
                    help='')
parser.add_argument('--draw_raw_tsne', action='store_true',
                    help='')
parser.add_argument('--use_tensorboard', action='store_true',
                    help='')
parser.add_argument('--use_gpu_valid', action='store_true',
                    help='')
parser.add_argument('--profile_timing', action='store_true',
                    help='')
parser.add_argument('--dataloader_num_workers', default=0, type=int)
parser.add_argument('--result_csv', default='results/param_results.csv', type=str,
                    help='')
parser.add_argument('--no_save_csv', action='store_true',
                    help='')
parser.add_argument('--exp_group', default='param_exp', type=str,
                    help='')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def set_current_param(name, value):
    args.param_name = name
    args.param_value = value


def set_arg(dataset_name):
    args.dataset = dataset_name
    if dataset_name == "MNIST-USPS":
        args.num_users = 24
        args.main_epochs = 100
        args.interval_epoch = 5
        args.lay_nums = [6, 6]
        args.l1_factor = 0
        args.l2_factor = 0
        args.noise_std = 0.1
        args.mask_rates = []
        args.feature_dim = 20
    if dataset_name == "BDGP":
        args.num_users = 12
        args.main_epochs = 70
        args.interval_epoch = 5
        args.lay_nums = [7, 2]
        args.l1_factor = 0
        args.l2_factor = 1
        args.noise_std = 0.1
        args.mask_rates = []
        args.feature_dim = 20
    if dataset_name == "Fashion":
        args.num_users = 48
        args.main_epochs = 120
        args.interval_epoch = 5
        args.lay_nums = [6, 6, 6]
        args.l1_factor = 0
        args.l2_factor = 0
        args.noise_std = 0.1
        args.mask_rates = [[2, 2, 1]]
        args.feature_dim = 20
    if dataset_name == "NUSWIDE":
        args.num_users = 24
        args.main_epochs = 100
        args.interval_epoch = 5
        args.lay_nums = [2, 4, 3, 2, 3]
        args.l1_factor = 0
        args.l2_factor = 0
        args.noise_std = 0.1
        args.mask_rates = [[1, 1, 1, 1, 1]]
        args.feature_dim = 20

    args.mask_rate = 0
    args.missing_rate = 0.5
    args.sample_num = 0
    args.participate = 1
    args.ablation_index = 0
    args.batch_size = 4096
    args.temperature_f = 0.5


if __name__ == '__main__':
    img_dir = 'images'
    if os.path.exists(img_dir):
        print("delete_files_in_directory(img_dir)")
    else:
        os.mkdir(img_dir)
    print(f"{img_dir} not exists")

    log_dir = 'runs'
    if os.path.exists(log_dir):
        print("delete_files_in_directory(log_dir)")
        time.sleep(1)
    print(f"{log_dir} not exists")

    set_arg('Fashion')
    set_current_param('pre_run', 'Fashion_default')
    train(args, 0, device)
    clear_cache()

    dataset_list = ['MNIST-USPS', 'BDGP', 'Fashion', 'NUSWIDE']
    for dataset_name in dataset_list:
        set_arg(dataset_name)
        css = [8192, 4096, 2048, 1024, 512]
        for cs in css:
            args.batch_size = cs
            args.missing_rate = 0.5
            set_current_param('batch_size', cs)
            for experiment_epo in range(args.repeated_experiment_epoch):
                train(args, experiment_epo, device)
                clear_cache()

    for dataset_name in dataset_list:
        set_arg(dataset_name)
        css = [1, 3, 5, 7, 9]
        for cs in css:
            args.interval_epoch = cs
            args.missing_rate = 0.5
            set_current_param('interval_epoch', cs)
            for experiment_epo in range(args.repeated_experiment_epoch):
                train(args, experiment_epo, device)
                clear_cache()

    for dataset_name in dataset_list:
        set_arg(dataset_name)
        css = [0.1, 0.3, 0.5, 0.7, 0.9]
        for cs in css:
            args.l2_factor = cs
            args.missing_rate = 0.5
            set_current_param('l2_factor', cs)
            for experiment_epo in range(args.repeated_experiment_epoch):
                train(args, experiment_epo, device)
                clear_cache()

    for dataset_name in dataset_list:
        set_arg(dataset_name)
        css = [0.1, 0.3, 0.5, 0.7, 0.9]
        for cs in css:
            args.temperature_f = cs
            args.missing_rate = 0.5
            set_current_param('temperature_f', cs)
            for experiment_epo in range(args.repeated_experiment_epoch):
                train(args, experiment_epo, device)
                clear_cache()
