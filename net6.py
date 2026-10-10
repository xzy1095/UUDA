import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out


class Network6(nn.Module):
    def __init__(self, total_view_num, dims, feature_dim, args, have_view, device):
        super(Network6, self).__init__()
        self.args = args
        self.have_view = have_view
        self.device = device
        self.init_center = None
        self.total_view_num = total_view_num
        self.register_buffer("view_eye", torch.eye(total_view_num, dtype=torch.float32))

        self.w_net = nn.Sequential(
            nn.Linear(2 * feature_dim + 3 * total_view_num, feature_dim + total_view_num),
            nn.ReLU(),
            nn.Linear(feature_dim + total_view_num, 1)
        ).to(self.device)
        self.af_net = nn.Sequential(
            nn.Linear(2 * feature_dim + 3 * total_view_num, feature_dim),
        ).to(self.device)
        self.fw_net = nn.Sequential(
            nn.Linear(feature_dim + 2 * total_view_num, feature_dim + total_view_num),
            nn.ReLU(),
            nn.Linear(feature_dim + total_view_num, feature_dim)
        ).to(self.device)
        self.w_net2 = nn.Sequential(
            nn.Linear(feature_dim + 2 * total_view_num, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        ).to(self.device)
        self.w_net3 = nn.Sequential(
            nn.Linear(2 * total_view_num, total_view_num),
            nn.ReLU(),
            nn.Linear(total_view_num, 1),
            nn.Sigmoid()
        ).to(self.device)
        self.w_net4 = nn.Sequential(
            nn.Linear(2 * total_view_num, 3 * total_view_num),
            nn.ReLU(),
            nn.Linear(3 * total_view_num, feature_dim),
            nn.Sigmoid()
        ).to(self.device)
        self.w_net5 = nn.Sequential(
            nn.Linear(feature_dim + 2 * total_view_num, feature_dim + total_view_num),
            nn.ReLU(),
            nn.Linear(feature_dim + total_view_num, feature_dim),
            nn.Sigmoid()
        ).to(self.device)
        self.w_net6 = nn.Sequential(
            nn.Linear(total_view_num, total_view_num),
            nn.ReLU(),
            nn.Linear(total_view_num, total_view_num),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.trust_net = nn.Sequential(
            nn.Linear(feature_dim + 2 * total_view_num, 256),
            nn.ReLU(),
            nn.Linear(256, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.Sigmoid()
        ).to(self.device)

        encoders = []
        decoders = []
        h_nets = []
        for v in range(total_view_num):
            encoders.append(nn.Sequential(
                nn.Linear(dims[v], 32 * feature_dim),
                nn.ReLU(),
                nn.Linear(32 * feature_dim, 16 * feature_dim),
                nn.ReLU(),
                nn.Linear(16 * feature_dim, 8 * feature_dim),
                nn.ReLU(),
                nn.Linear(8 * feature_dim, 4 * feature_dim),
                nn.ReLU(),
                nn.Linear(4 * feature_dim, 2 * feature_dim),
                nn.ReLU(),
                nn.Linear(2 * feature_dim, feature_dim)
            ).to(self.device))
            decoders.append(nn.Sequential(
                nn.Linear(feature_dim, 2 * feature_dim),
                nn.ReLU(),
                nn.Linear(2 * feature_dim, 4 * feature_dim),
                nn.ReLU(),
                nn.Linear(4 * feature_dim, 8 * feature_dim),
                nn.ReLU(),
                nn.Linear(8 * feature_dim, 16 * feature_dim),
                nn.ReLU(),
                nn.Linear(16 * feature_dim, 32 * feature_dim),
                nn.ReLU(),
                nn.Linear(32 * feature_dim, dims[v])
            ).to(self.device))
            h_nets.append(nn.Sequential(
                ResidualBlock(feature_dim, feature_dim),
                ResidualBlock(feature_dim, feature_dim)
            ).to(self.device))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(copy.deepcopy(decoders))
        self.v_decoders = nn.ModuleList(copy.deepcopy(decoders))
        self.h_nets = nn.ModuleList(h_nets)

    def forward(self, xs, view_code):
        # 保持原逻辑：所有 view 都编码，包括缺失 view 的 0 向量。
        # 只做浅拷贝，避免修改调用方传入的 xs 列表内容。
        p1 = list(xs)
        f_list = [None] * self.total_view_num
        batch_size = xs[0].shape[0]

        va = torch.zeros(self.total_view_num, device=self.device, dtype=xs[0].dtype)
        va[view_code] = 1
        vas = va.unsqueeze(0).expand(batch_size, -1)

        for v in range(self.total_view_num):
            p1[v] = self.encoders[v](p1[v])
            f_list[v] = F.normalize(p1[v], dim=1)

        if self.args.ablation_index != 1:
            weighted_sum = torch.zeros_like(f_list[view_code[0]])
            trust_sum = torch.zeros_like(f_list[view_code[0]])
            for i in view_code:
                codes_i = self.view_eye[i].to(dtype=xs[0].dtype).unsqueeze(0).expand(batch_size, -1)
                trust = self.trust_net(torch.cat((f_list[i], codes_i, vas), dim=1))
                weighted_sum = weighted_sum + trust * f_list[i]
                trust_sum = trust_sum + trust
            common_f = weighted_sum / trust_sum
        else:
            common_f = f_list[random.choice(view_code)]

        # 用初始特征还原出样本所有模态；去掉 clone/to，不改变计算图含义。
        xrs = [None] * self.total_view_num
        for v in range(self.total_view_num):
            xrs[v] = self.v_decoders[v](f_list[v])

        xrrs = [None] * self.total_view_num
        for v in range(self.total_view_num):
            xrrs[v] = self.decoders[v](common_f)

        hs = [None] * self.total_view_num
        for v in range(self.total_view_num):
            hs[v] = self.h_nets[v](f_list[v])

        return f_list, common_f, xrrs, xrs, hs
