import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import DatasetSplit, get_mask2, get_mask3, load_data
from loss import my_loss101, my_loss102
from net6 import Network6
from utils import all_combinations, aggregate_models, draw_scatter, get_models_weights_list, valid_list, valid_list_gpu


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _get_arg(args, name, default):
    return getattr(args, name, default)


def _make_loader(dataset_x, dataset_y, idxs, dims, views, batch_size, shuffle, pin_memory, num_workers):
    persistent_workers = bool(num_workers and num_workers > 0)
    return DataLoader(
        DatasetSplit(dataset_x, dataset_y, idxs, dims, views),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def _move_xs_to_device(xs, device):
    return [x.to(device, dtype=torch.float32, non_blocking=True) for x in xs]


def _should_run_eval(glob_epo, main_epochs, eval_interval):
    """
    Communication round is recorded in 1-based form: 1, 2, ..., main_epochs.
    eval_interval=10 means evaluate/save after rounds 10, 20, 30, ... and always after the final round.
    eval_interval<=0 means only evaluate/save after the final round.
    """
    comm_round = glob_epo + 1
    if comm_round == main_epochs:
        return True
    if eval_interval is None or eval_interval <= 0:
        return False
    return comm_round % eval_interval == 0


def _safe_json(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _append_csv_result(args, row):
    """Append one evaluation result to CSV immediately.

    This function only records results; it does not change training, aggregation, loss, or evaluation logic.
    Disable by passing --no_save_csv or by setting --result_csv ''.
    """
    if bool(getattr(args, "no_save_csv", False)):
        return
    csv_path = str(getattr(args, "result_csv", "results/param_results.csv"))
    if csv_path.strip() == "":
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = [
        "timestamp",
        "exp_group",
        "dataset",
        "param_name",
        "param_value",
        "experiment_epo",
        "comm_round",
        "epoch_index",
        "acc",
        "nmi",
        "ari",
        "main_epochs",
        "interval_epoch",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "temperature_f",
        "l1_factor",
        "l2_factor",
        "noise_std",
        "missing_rate",
        "mask_rate",
        "mask_rates",
        "sample_num",
        "num_users",
        "active_users",
        "participate",
        "ablation_index",
        "Dirichlet_alpha",
        "feature_dim",
        "total_view_num",
        "class_num",
        "eval_interval",
        "use_gpu_valid",
        "elapsed_train_sec",
        "elapsed_agg_weight_sec",
        "elapsed_eval_sec",
    ]
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


def train(args, experiment_epo, device):
    res = 0

    dataset, dims, total_view_num, data_size, class_num = load_data(
        args.dataset, args.num_users, args.Dirichlet_alpha
    )

    pin_memory = bool(device.type == "cuda")
    num_workers = int(_get_arg(args, "dataloader_num_workers", 0))
    eval_interval = int(_get_arg(args, "eval_interval", 1))
    draw_raw_tsne = bool(_get_arg(args, "draw_raw_tsne", False))
    use_tensorboard = bool(_get_arg(args, "use_tensorboard", False))
    use_gpu_valid = bool(_get_arg(args, "use_gpu_valid", False))
    profile_timing = bool(_get_arg(args, "profile_timing", False))

    if args.missing_rate > -0.1:
        num_views_glob = get_mask2(total_view_num, args.num_users, args.missing_rate)
    else:
        num_views_glob = get_mask3(args.num_users, args.mask_rate)

    data_loader_list = []
    test_data_loader_list = []
    models_weights_list = []

    for j in range(args.num_users):
        if args.sample_num > 0.01:
            dataset.user_data[j] = dataset.user_data[j][:args.sample_num]
        data_loader = _make_loader(
            dataset.X, dataset.Y, dataset.user_data[j], dims, num_views_glob[j],
            batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers,
        )
        data_loader_list.append(data_loader)
        models_weights_list.append(len(num_views_glob[j]))

    for j in range(args.num_users):
        test_data_loader = _make_loader(
            dataset.X, dataset.Y, dataset.user_data[j], dims, num_views_glob[j],
            batch_size=data_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers,
        )
        test_data_loader_list.append(test_data_loader)

    print('num_views_glob', num_views_glob)

    name = (
        str(args.dataset) + 'nu' + str(args.num_users) + 'missr' + str(args.missing_rate)
        + 'maskr' + str(args.mask_rate) + '_l1_' + str(args.l1_factor)
        + '_l2_' + str(args.l2_factor) + '_sn' + str(args.sample_num)
        + 'p' + str(args.participate) + 'tf' + str(args.temperature_f)
        + 'ns' + str(args.noise_std) + 'ai' + str(args.ablation_index)
    )
    log_dir = 'runs/' + name + '/' + str(experiment_epo)
    writer = SummaryWriter(log_dir) if use_tensorboard else NullWriter()

    print(log_dir + '   Start Training')

    active_users = int(args.num_users * args.participate)
    local_models = []
    for j in range(active_users):
        have_view = num_views_glob[j]
        local_models.append(Network6(total_view_num, dims, args.feature_dim, args, have_view, device).to(device))

    # 预先计算每个 client 的 masked_views，保持原逻辑：只保留出现在 num_views_glob 中的 subset。
    masked_views_all = []
    for nu in range(active_users):
        have_view = num_views_glob[nu]
        combinations_views = all_combinations(have_view)
        masked_views = []
        for vs in combinations_views:
            if vs in num_views_glob:
                masked_views.append(vs)
        masked_views_all.append(masked_views)
        # combinations_views = all_combinations(have_view)
        # masked_views_all.append(combinations_views)

    mes = torch.nn.MSELoss()
    first_train = True

    # 原代码每次 train 开头都画 raw t-SNE，参数实验中这是主要 CPU 阻塞之一。
    # 默认关闭；需要复现绘图时设置 args.draw_raw_tsne=True。
    if draw_raw_tsne:
        for batch_idx, (xs, ys) in enumerate(data_loader_list[0]):
            for v in range(len(xs)):
                draw_scatter(xs[v], ys, name + '_raw_data_' + str(v) + '_' + str(batch_idx))

    for glob_epo in range(args.main_epochs):
        if profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for nu in range(active_users):
            have_view = num_views_glob[nu]
            masked_views = masked_views_all[nu]
            model = local_models[nu]

            # 保持原有训练逻辑：每个 global epoch、每个 client 重新创建 Adam。
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            model.train()

            for pre_epoch in range(args.interval_epoch):
                tot_loss = 0.0
                for masked_view in masked_views:
                    for batch_idx, (xs, ys) in enumerate(data_loader_list[nu]):
                        xs = _move_xs_to_device(xs, device)
                        optimizer.zero_grad()

                        noise_xs = []
                        for v_xs in xs:
                            noise_xs.append(v_xs + torch.randn_like(v_xs) * args.noise_std)

                        f_list, common_f, xrrs, xrs, hs = model(noise_xs, masked_view)
                        loss_list = []
                        for v in have_view:
                            if args.ablation_index != 3:
                                loss_list.append(
                                    my_loss101(noise_xs[v], xrrs[v], xrs[v], args.temperature_f, args.l1_factor)
                                )
                            else:
                                loss_list.append(
                                    mes(noise_xs[v], xrs[v]) + mes(noise_xs[v], xrrs[v]) + mes(xrs[v], xrrs[v])
                                )
                            if args.ablation_index != 4:
                                loss_list.append(my_loss102(f_list[v], common_f, args.temperature_f, args.l2_factor))
                        loss = sum(loss_list) / len(have_view)
                        loss.backward()
                        optimizer.step()
                        tot_loss += loss.item()

            if first_train:
                for m in local_models:
                    m.load_state_dict(model.state_dict())
            first_train = False

        if profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        if args.ablation_index != 2:
            models_weights_list = get_models_weights_list(
                local_models, test_data_loader_list, active_users, class_num, device
            )
            global_model = aggregate_models(local_models, weights=models_weights_list)
            for m in local_models:
                m.load_state_dict(global_model.state_dict())

        if profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()

        eval_time = 0.0
        if _should_run_eval(glob_epo, args.main_epochs, eval_interval):
            if profile_timing and device.type == "cuda":
                torch.cuda.synchronize()
            t_eval0 = time.time()

            if use_gpu_valid:
                acc, nmi, ari, init_center = valid_list_gpu(
                    local_models, test_data_loader_list, active_users, class_num, device
                )
            else:
                acc, nmi, ari, init_center = valid_list(
                    local_models, test_data_loader_list, active_users, class_num, device
                )

            if profile_timing and device.type == "cuda":
                torch.cuda.synchronize()
            t_eval1 = time.time()
            eval_time = t_eval1 - t_eval0

            for m in local_models:
                m.init_center = init_center
            writer.add_scalar('overall_train/acc', acc, glob_epo)
            writer.add_scalar('overall_train/nmi', nmi, glob_epo)
            writer.add_scalar('overall_train/ari', ari, glob_epo)
            res = acc

            comm_round = glob_epo + 1
            _append_csv_result(args, {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "exp_group": getattr(args, "exp_group", "param_exp"),
                "dataset": args.dataset,
                "param_name": getattr(args, "param_name", "default"),
                "param_value": getattr(args, "param_value", "default"),
                "experiment_epo": experiment_epo,
                "comm_round": comm_round,
                "epoch_index": glob_epo,
                "acc": float(acc),
                "nmi": float(nmi),
                "ari": float(ari),
                "main_epochs": int(args.main_epochs),
                "interval_epoch": int(args.interval_epoch),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "temperature_f": float(args.temperature_f),
                "l1_factor": float(args.l1_factor),
                "l2_factor": float(args.l2_factor),
                "noise_std": float(args.noise_std),
                "missing_rate": float(args.missing_rate),
                "mask_rate": _safe_json(args.mask_rate),
                "mask_rates": _safe_json(getattr(args, "mask_rates", [])),
                "sample_num": int(args.sample_num),
                "num_users": int(args.num_users),
                "active_users": int(active_users),
                "participate": float(args.participate),
                "ablation_index": int(args.ablation_index),
                "Dirichlet_alpha": float(args.Dirichlet_alpha),
                "feature_dim": int(args.feature_dim),
                "total_view_num": int(total_view_num),
                "class_num": int(class_num),
                "eval_interval": int(eval_interval),
                "use_gpu_valid": bool(use_gpu_valid),
                "elapsed_train_sec": float(t1 - t0),
                "elapsed_agg_weight_sec": float(t2 - t1),
                "elapsed_eval_sec": float(eval_time),
            })

            print(
                f"epoch={glob_epo:04d} round={comm_round:04d} "
                f"acc={acc:.4f} nmi={nmi:.4f} ari={ari:.4f} "
                f"csv={getattr(args, 'result_csv', 'results/param_results.csv')}"
            )

        if profile_timing:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.time()
            print(
                f"[timing] epoch={glob_epo:04d} train={t1 - t0:.2f}s "
                f"agg_weight={t2 - t1:.2f}s eval={eval_time:.2f}s total_after_agg={t3 - t2:.2f}s"
            )

    writer.close()
    return res
