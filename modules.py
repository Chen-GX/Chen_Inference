import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class Encoder(nn.Module):
    def __init__(self, n_in, n_hid1, n_out, device, num_cascade):
        super(Encoder, self).__init__()
        self.device = device
        self.mlp1_out = n_hid1[1]
        # self.linears为每个cascade构建一个三层MLP
        self.linears = nn.ModuleList([MLP(n_in, n_hid1[0], n_hid1[1]) for _ in range(num_cascade)])
        self.linear2z = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_hid1[1], n_out),
            nn.ReLU()
        )

    def forward(self, input):
        """

        :param input: [node_id, cascade_num, feature_dim]
        :return:
        """
        dim0, dim1, dim2 = input.shape
        y = torch.zeros(dim0, self.mlp1_out).to(self.device)  # [batch_num_nodes, n_hid1[1]]
        for i in range(dim0):  # 遍历节点
            for j in range(dim1):  # 遍历cascade
                y[i] += self.linears[j](input[i][j])  # 每个cascade有单独的MLP
        y /= dim1
        z = self.linear2z(y)
        return z


class Decoder(nn.Module):
    def __init__(self, n_in, n_hid1, n_out, device, num_cascade):
        super(Decoder, self).__init__()
        self.y_dim = n_hid1[0]  # z转化为y的维度
        self.device = device
        self.num_cascade = num_cascade
        self.n_out = n_out
        self.linear2y = nn.Sequential(
            nn.Linear(n_in, n_hid1[0]),
            nn.ReLU()
        )
        # self.linears为每个cascade构建一个三层MLP  但是少最后一个relu
        self.linears = nn.ModuleList([MLP(n_hid1[0], n_hid1[1], n_out) for _ in range(num_cascade)])

    def forward(self, input):
        dim0, dim1 = input.shape  # [node_id, z_dim]
        x_hat = torch.zeros(dim0, self.num_cascade, self.n_out).to(self.device)
        for i in range(dim0):  # 遍历batch中的节点
            y_i = self.linear2y(input[i])
            for j in range(self.num_cascade):
                x_hat[i][j] = F.relu(self.linears[j](y_i))
        return x_hat


