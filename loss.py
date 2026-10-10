import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_similarity_matrix(matrix1, matrix2):
    norm1 = torch.norm(matrix1, p=2, dim=1, keepdim=True)  # 每行的L2范数
    norm2 = torch.norm(matrix2, p=2, dim=1, keepdim=True)  # 每行的L2范数

    normalized_matrix1 = matrix1 / norm1  # 每行除以自身的L2范数
    normalized_matrix2 = matrix2 / norm2  # 每行除以自身的L2范数

    similarity_matrix = torch.matmul(normalized_matrix1, normalized_matrix2.T)  # 矩阵乘法
    return similarity_matrix


def my_loss(matrix_v, matrix_g, temperature_f):
    similarity_matrix = compute_similarity_matrix(matrix_v, matrix_g)
    similarity_matrix = torch.exp(similarity_matrix / temperature_f)
    # print(similarity_matrix)
    diagonal_elements = torch.diag(similarity_matrix)
    # print(diagonal_elements)
    tensor_0 = torch.sum(similarity_matrix, dim=0) - diagonal_elements
    tensor_1 = torch.sum(similarity_matrix, dim=1) - diagonal_elements
    # print(tensor_0)
    # print(tensor_1)
    loss_tensor = torch.log(diagonal_elements / (tensor_0 + tensor_1))
    # print(loss_tensor)
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss2(feature_v, frature_g, xs, xrrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    feature_similarity_matrix = compute_similarity_matrix(feature_v, frature_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)
    tensor_0 = torch.sum(feature_similarity_matrix, dim=0) - feature_diagonal_elements
    tensor_1 = torch.sum(feature_similarity_matrix, dim=1) - feature_diagonal_elements
    loss_tensor = torch.log((feature_diagonal_elements + 0.1 * x_diagonal_elements) / (tensor_0 + tensor_1))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss3(f_v, f_g, h_v, xs, xrrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    hvfg_similarity_matrix = compute_similarity_matrix(h_v, f_g)
    hvfg_similarity_matrix = torch.exp(hvfg_similarity_matrix / temperature_f)
    hvfg_diagonal_elements = torch.diag(hvfg_similarity_matrix)

    hvfv_similarity_matrix = compute_similarity_matrix(h_v, f_v)
    hvfv_similarity_matrix = torch.exp(hvfv_similarity_matrix / temperature_f)
    hvfv_diagonal_elements = torch.diag(hvfv_similarity_matrix)

    hvhv_similarity_matrix = compute_similarity_matrix(h_v, f_v)
    hvhv_similarity_matrix = torch.exp(hvhv_similarity_matrix / temperature_f)
    hvhv_diagonal_elements = torch.diag(hvhv_similarity_matrix)
    tensor_0 = torch.sum(hvhv_similarity_matrix, dim=0) - hvhv_diagonal_elements
    tensor_1 = torch.sum(hvhv_similarity_matrix, dim=1) - hvhv_diagonal_elements

    loss_tensor = torch.log(
        (x_diagonal_elements + hvfg_diagonal_elements + hvfv_diagonal_elements) / (tensor_0 + tensor_1))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss4(feature_v, frature_g, xs, xrrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    feature_similarity_matrix = compute_similarity_matrix(feature_v, frature_g)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)

    loss_tensor = feature_diagonal_elements * x_diagonal_elements
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss5(feature_v, frature_g, xs, xrrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    feature_similarity_matrix = compute_similarity_matrix(feature_v, frature_g)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)

    tensor_0 = torch.sum(feature_similarity_matrix, dim=0) - feature_diagonal_elements
    tensor_1 = torch.sum(feature_similarity_matrix, dim=1) - feature_diagonal_elements

    loss_tensor = feature_diagonal_elements * x_diagonal_elements / (tensor_0 + tensor_1)
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def sim(a, b):
    similarity_matrix = compute_similarity_matrix(a, b)
    diagonal_elements = torch.diag(similarity_matrix)
    return diagonal_elements

def sim_loss(a, b):
    loss_tensor = sim(a, b)
    return torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss6(feature_v, frature_g, xs, xrrs, xrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xs, xrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    feature_similarity_matrix = compute_similarity_matrix(feature_v, frature_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)
    tensor_0 = torch.sum(feature_similarity_matrix, dim=0) - feature_diagonal_elements
    tensor_1 = torch.sum(feature_similarity_matrix, dim=1) - feature_diagonal_elements
    loss_tensor = torch.log((feature_diagonal_elements * x_diagonal_elements * x_diagonal_elements2) / (tensor_0 + tensor_1))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss7(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xrs, xrrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    hvfg_similarity_matrix = compute_similarity_matrix(h_v, f_g)
    hvfg_similarity_matrix = torch.exp(hvfg_similarity_matrix / temperature_f)
    hvfg_diagonal_elements = torch.diag(hvfg_similarity_matrix)

    hvfv_similarity_matrix = compute_similarity_matrix(h_v, f_v)
    hvfv_similarity_matrix = torch.exp(hvfv_similarity_matrix / temperature_f)
    hvfv_diagonal_elements = torch.diag(hvfv_similarity_matrix)

    hvhv_similarity_matrix = compute_similarity_matrix(h_v, h_v)
    hvhv_similarity_matrix = torch.exp(hvhv_similarity_matrix / temperature_f)
    hvhv_diagonal_elements = torch.diag(hvhv_similarity_matrix)
    tensor_0 = torch.sum(hvhv_similarity_matrix, dim=0) - hvhv_diagonal_elements
    tensor_1 = torch.sum(hvhv_similarity_matrix, dim=1) - hvhv_diagonal_elements

    loss_tensor = torch.log(
        x_diagonal_elements + x_diagonal_elements2 + (hvfg_diagonal_elements + hvfv_diagonal_elements) / (tensor_0 + tensor_1))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss8(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xrs, xrrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)
    tensor_0 = torch.sum(feature_similarity_matrix, dim=0) - feature_diagonal_elements
    tensor_1 = torch.sum(feature_similarity_matrix, dim=1) - feature_diagonal_elements

    loss_tensor = torch.log(
        x_diagonal_elements + x_diagonal_elements2 + feature_diagonal_elements / (tensor_0 + tensor_1))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss9(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xs, xrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)
    tensor_0 = torch.sum(feature_similarity_matrix, dim=0) - feature_diagonal_elements
    tensor_1 = torch.sum(feature_similarity_matrix, dim=1) - feature_diagonal_elements

    loss_tensor = torch.log(
        x_diagonal_elements + x_diagonal_elements2 + feature_diagonal_elements / (tensor_0 + tensor_1))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]

def my_loss9(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xs, xrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)
    tensor_0 = torch.sum(feature_similarity_matrix, dim=0) - feature_diagonal_elements
    tensor_1 = torch.sum(feature_similarity_matrix, dim=1) - feature_diagonal_elements

    loss_tensor = torch.log((x_diagonal_elements + x_diagonal_elements2) + feature_diagonal_elements / (tensor_0 + tensor_1))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss10(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f):
    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)
    tensor_0 = torch.sum(feature_similarity_matrix, dim=0) - feature_diagonal_elements
    tensor_1 = torch.sum(feature_similarity_matrix, dim=1) - feature_diagonal_elements

    loss_tensor = torch.log(feature_diagonal_elements / (tensor_0 + tensor_1))

    mae_loss_tensor_0 = torch.mean(torch.abs(xs - xrrs), dim=1)
    mae_loss_tensor_1 = torch.mean(torch.abs(xs - xrs), dim=1)

    return - torch.sum(mae_loss_tensor_0 + mae_loss_tensor_1 + loss_tensor) / loss_tensor.shape[0]


def my_loss11(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xs, xrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)

    f_v_similarity_matrix = compute_similarity_matrix(f_v, f_v)
    f_v_similarity_matrix = torch.exp(f_v_similarity_matrix / temperature_f)
    f_v_diagonal_elements = torch.diag(f_v_similarity_matrix)
    tensor_0 = (torch.sum(f_v_similarity_matrix, dim=0) - f_v_diagonal_elements) / 2

    loss_tensor = torch.log(x_diagonal_elements + x_diagonal_elements2 + feature_diagonal_elements)
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def compute_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    absolute_errors = torch.abs(y_pred - y_true)
    return torch.mean(absolute_errors)


def my_loss12(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f, l1_factor, l2_factor):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xs, xrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    x_similarity_matrix3 = compute_similarity_matrix(xrs, xrrs)
    x_similarity_matrix3 = torch.exp(x_similarity_matrix3 / temperature_f)
    x_diagonal_elements3 = torch.diag(x_similarity_matrix3)

    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)

    f_v_similarity_matrix = compute_similarity_matrix(f_v, f_v)
    f_v_similarity_matrix = torch.exp(f_v_similarity_matrix / temperature_f)
    f_v_diagonal_elements = torch.diag(f_v_similarity_matrix)
    tensor_0 = (torch.sum(f_v_similarity_matrix, dim=0) - f_v_diagonal_elements) / 2

    loss_tensor = torch.log((l1_factor + x_diagonal_elements + x_diagonal_elements2 + x_diagonal_elements3) * (l2_factor + feature_diagonal_elements / tensor_0))
    # loss_tensor = torch.log((1 + x_diagonal_elements + x_diagonal_elements2 + x_diagonal_elements3) * (feature_diagonal_elements / tensor_0))
    # loss_tensor = torch.log((l1_factor + x_diagonal_elements + x_diagonal_elements2 + x_diagonal_elements3) * feature_diagonal_elements / (tensor_0 + feature_diagonal_elements))
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss13(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xs, xrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)

    f_v_similarity_matrix = compute_similarity_matrix(f_v, f_v)
    f_v_similarity_matrix = torch.exp(f_v_similarity_matrix / temperature_f)
    f_v_diagonal_elements = torch.diag(f_v_similarity_matrix)
    tensor_0 = (torch.sum(f_v_similarity_matrix, dim=0) - f_v_diagonal_elements) / 2

    mae_loss_tensor_0 = torch.mean(torch.abs(xs - xrrs), dim=1)
    mae_loss_tensor_1 = torch.mean(torch.abs(xs - xrs), dim=1)

    loss_tensor = mae_loss_tensor_0 ** 2 + mae_loss_tensor_1 ** 2
    return torch.sum(loss_tensor) / loss_tensor.shape[0]


def my_loss14(f_v, f_g, h_v, xs, xrrs, xrs, temperature_f, l2_factor):
    x_similarity_matrix = compute_similarity_matrix(xs, xrrs)
    x_similarity_matrix = torch.exp(x_similarity_matrix / temperature_f)
    x_diagonal_elements = torch.diag(x_similarity_matrix)

    x_similarity_matrix2 = compute_similarity_matrix(xs, xrs)
    x_similarity_matrix2 = torch.exp(x_similarity_matrix2 / temperature_f)
    x_diagonal_elements2 = torch.diag(x_similarity_matrix2)

    feature_similarity_matrix = compute_similarity_matrix(f_v, f_g)
    feature_similarity_matrix = torch.exp(feature_similarity_matrix / temperature_f)
    feature_diagonal_elements = torch.diag(feature_similarity_matrix)

    f_v_similarity_matrix = compute_similarity_matrix(f_v, f_v)
    f_v_similarity_matrix = torch.exp(f_v_similarity_matrix / temperature_f)
    f_v_diagonal_elements = torch.diag(f_v_similarity_matrix)
    tensor_0 = (torch.sum(f_v_similarity_matrix, dim=0) - f_v_diagonal_elements) / 2

    loss_tensor = torch.log(x_diagonal_elements + x_diagonal_elements2 + feature_diagonal_elements / tensor_0)
    return - torch.sum(loss_tensor) / loss_tensor.shape[0]

def my_loss101(xs, xrrs, xrs, temperature_f, l1_factor):
    def f1(a, b):
        similarity_matrix = compute_similarity_matrix(a, b)
        similarity_matrix = torch.exp(similarity_matrix / temperature_f)
        diagonal_elements = torch.diag(similarity_matrix)
        diff_matrix = (torch.sum(similarity_matrix, dim=0) - diagonal_elements) / 2
        return diagonal_elements, diff_matrix

    x_e1, x_m1 = f1(xs, xrs)
    x_e2, x_m2 = f1(xs, xrrs)
    x_e3, x_m3 = f1(xrs, xrrs)

    loss_tensor = torch.log(l1_factor + x_e1 + x_e2 + x_e3)

    return - torch.sum(loss_tensor) / loss_tensor.shape[0]

def my_loss102(fv, cf, temperature_f, l2_factor):
    def f1(a, b):
        similarity_matrix = compute_similarity_matrix(a, b)
        similarity_matrix = torch.exp(similarity_matrix / temperature_f)
        diagonal_elements = torch.diag(similarity_matrix)
        diff_matrix = (torch.sum(similarity_matrix, dim=0) - diagonal_elements) / 2
        return diagonal_elements, diff_matrix

    fvcf_e, fvcf_m = f1(fv, cf)

    loss_tensor = torch.log(l2_factor + (fvcf_e) / (fvcf_m))

    return - torch.sum(loss_tensor) / loss_tensor.shape[0]