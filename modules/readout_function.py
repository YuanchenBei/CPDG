import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# 平均值readout
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, mat):
        return torch.mean(mat, 0)


# 最大值readout
class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, mat):
        val, idx = torch.max(mat, 0)
        return val


# 最小值readout
class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, mat):
        val, idx = torch.min(mat, 0)
        return val


# 带权重readout，这里先采用一个简单的线性变换
class WeightedAvgReadout(nn.Module):
    def __init__(self, emb_dim):
        super(WeightedAvgReadout, self).__init__()
        self.trans = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, mat):
        scores = F.softmax(self.trans(mat), dim=0)
        return torch.sum(scores*mat, 0)
