import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        """实现三层的MLP"""
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out),
        )

    def forward(self, input):
        x = self.mlp(input)
        return x
