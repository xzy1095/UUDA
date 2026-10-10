import copy
import math
from itertools import combinations

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import metric


def all_combinations(input_vector):
    result = []
    for r in range(1, len(input_vector) + 1):
        combos = combinations(input_vector, r)
        for combo in combos:
            result.append(sorted(list(combo), reverse=False))
    return result


def _move_xs_to_device(xs, device):
    return [x.to(device, dtype=torch.float32, non_blocking=True) for x in xs]


def get_models_weights_list(valid_model_list, valid_dataset_list, valid_users, class_num, device):
    models_weights_list = []
    with torch.no_grad():
        for an in range(valid_users):
            valid_model_list[an].eval()
            model = valid_model_list[an]
            mi = 0.0
            for batch_idx, (xs, ys) in enumerate(valid_dataset_list[an]):
                xs = _move_xs_to_device(xs, device)
                f_list, common_f, xrrs, xrs, hs = model(xs, model.have_view)
                for v in model.have_view:
                    mi += math.exp(-metric.mutual_information(f_list[v], common_f))
            models_weights_list.append(mi / len(model.have_view))
    return models_weights_list


def aggregate_models(model_list, weights=None):
    """保持原有聚合公式：sum(weighted local params) / sum(weights)。"""
    if weights is None:
        weights = [1.0 / len(model_list)] * len(model_list)
    weight_sum = float(sum(weights))
    agg_model = copy.deepcopy(model_list[0])
    with torch.no_grad():
        agg_state_dict = agg_model.state_dict()
        for key in agg_state_dict:
            agg_state_dict[key].zero_()
        for model, weight in zip(model_list, weights):
            model_state_dict = model.state_dict()
            coef = float(weight) / weight_sum
            for key in agg_state_dict:
                agg_state_dict[key].add_(model_state_dict[key], alpha=coef)
        agg_model.load_state_dict(agg_state_dict)
    return agg_model


def get_label_by_KMeans(model, dataset, class_num, device):
    h_list = []
    ys_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(dataset):
            xs = _move_xs_to_device(xs, device)
            f_list, common_f, xrrs, xrs, hs = model(xs, model.have_view)
            h_list.append(common_f)
            ys_list.append(ys)
    local_hs = torch.cat(h_list, dim=0)
    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
    feats_np = local_hs.detach().cpu().numpy()
    kmeans.fit(feats_np)
    labels = kmeans.predict(feats_np)
    return labels


def torch_kmeans(x, num_clusters, num_iters=100, init_centers=None):
    N, D = x.shape
    device = x.device
    if init_centers is None:
        indices = torch.randperm(N, device=device)[:num_clusters]
        centers = x[indices].clone()
    else:
        centers = init_centers.clone()

    for _ in range(num_iters):
        dists = torch.cdist(x, centers)
        labels = torch.argmin(dists, dim=1)
        centers.zero_()
        counts = torch.zeros(num_clusters, device=device)
        centers.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones(N, device=device))
        mask = counts > 0
        centers[mask] = centers[mask] / counts[mask].unsqueeze(1)
    return labels, centers


def valid_list_gpu(valid_model_list, valid_dataset_list, valid_users, class_num, device):
    all_preds = []
    all_trues = []
    init_centers = None

    for uid in range(valid_users):
        model = valid_model_list[uid].to(device).eval()
        feats_list = []
        labels_list = []
        with torch.no_grad():
            for xs, ys in valid_dataset_list[uid]:
                xs = _move_xs_to_device(xs, device)
                _, common_f, *_ = model(xs, model.have_view)
                feats_list.append(common_f)
                labels_list.append(ys.to(device, non_blocking=True))
        feats = torch.cat(feats_list, dim=0)
        trues = torch.cat(labels_list, dim=0)
        preds, centers = torch_kmeans(feats, class_num, num_iters=100, init_centers=init_centers)
        if uid == 0:
            init_centers = centers
        all_preds.append(preds)
        all_trues.append(trues)

    global_preds = torch.cat(all_preds, dim=0).detach().cpu().numpy()
    global_trues = torch.cat(all_trues, dim=0).detach().cpu().numpy()
    acc = compute_acc(global_preds, global_trues, class_num)
    nmi = normalized_mutual_info_score(global_preds, global_trues)
    ari = adjusted_rand_score(global_preds, global_trues)
    return acc, nmi, ari, init_centers


def valid_list(valid_model_list, valid_dataset_list, valid_users, class_num, device):
    """保持原有 sklearn KMeans 评估逻辑，只避免构图和重复 CPU 转换。"""
    local_ys = []
    local_labels = []
    init_center = []
    with torch.no_grad():
        for an in range(valid_users):
            h_list = []
            ys_list = []
            valid_model_list[an].eval()
            model = valid_model_list[an]
            for batch_idx, (xs, ys) in enumerate(valid_dataset_list[an]):
                xs = _move_xs_to_device(xs, device)
                f_list, common_f, xrrs, xrs, hs = model(xs, model.have_view)
                h_list.append(common_f)
                ys_list.append(ys)
            local_hs = torch.cat(h_list, dim=0)
            local_ys.append(torch.cat(ys_list, dim=0))
            feats_np = local_hs.detach().cpu().numpy()

            if an == 0:
                kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
                kmeans.fit(feats_np)
                init_center = kmeans.cluster_centers_
            else:
                kmeans = KMeans(n_clusters=class_num, init=init_center)
                kmeans.fit(feats_np)
            labels = kmeans.predict(feats_np)
            local_labels.append(labels)

    global_ys = torch.cat(local_ys, dim=0)
    global_labels = np.concatenate(local_labels, axis=0)
    y_np = global_ys.detach().cpu().numpy()
    test_acc = compute_acc(global_labels, y_np, class_num)
    test_nmi = normalized_mutual_info_score(global_labels, y_np)
    test_ari = adjusted_rand_score(global_labels, y_np)
    return test_acc, test_nmi, test_ari, init_center


def valid_model(model, valid_dataset, nu, class_num, device):
    h_list = []
    ys_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(valid_dataset):
            xs = _move_xs_to_device(xs, device)
            f_list, common_f, xrrs, xrs, hs = model(xs, model.have_view)
            h_list.append(common_f)
            ys_list.append(ys)
    local_hs = torch.cat(h_list, dim=0)
    local_ys = torch.cat(ys_list, dim=0)
    feats_np = local_hs.detach().cpu().numpy()

    if model.init_center is None:
        kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
        kmeans.fit(feats_np)
        init_center = kmeans.cluster_centers_
        model.init_center = init_center
    else:
        kmeans = KMeans(n_clusters=class_num, init=model.init_center)
        kmeans.fit(feats_np)
    labels = kmeans.predict(feats_np)
    acc = compute_acc(labels, local_ys.detach().cpu().numpy(), class_num)
    return acc


def compute_acc(y_pred, y_true, class_num):
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def draw_scatter(data_tensor, labels, title):
    tsne = TSNE(n_components=2)
    data_2d = tsne.fit_transform(data_tensor.cpu().detach().numpy())
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10')
    plt.savefig('./images/{}.png'.format(title))
    plt.close()
