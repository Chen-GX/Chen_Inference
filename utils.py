import os
import numpy as np
import torch
from torch.utils.data import dataset, DataLoader
from torch.utils.data.dataset import TensorDataset


def load_data(args):
    data_path = os.path.join(args.data_path, args.data_log)
    X = np.load(data_path + "_X.npy")  # [node_id, cascade_num, feature_dim]
    occ = np.load(data_path + "_occ.npy")  # [node, node]
    net = np.load(data_path + "_net.npy")  # [node, node]]
    node_id = np.arange(X.shape[0])  # X.shape是个属性，返回一个元组，用[]提取需要的维度
    args.num_cascade = X.shape[1]
    # 打乱数据集 X 和
    X = torch.tensor(X, dtype=torch.float32)
    occ = torch.tensor(occ, dtype=torch.float32)
    node_id = torch.tensor(node_id, dtype=torch.int64)

    data = TensorDataset(X, node_id)

    data_loader = DataLoader(data, batch_size=args.batch_size)
    return data_loader, occ


def loss_function(x, x_hat, beta):
    rou = torch.ones_like(x) * beta
    e = torch.ones_like(x)
    p = torch.where(x == 0, rou, e)
    loss = ((x - x_hat).mul(p)) ** 2
    return torch.sum(loss)
