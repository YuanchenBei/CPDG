import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(2022)

# 将预训练任务与下游任务的embedding映射到同一对比空间
class DistributionMLP(nn.Module):
    def __init__(self, source_dim, hid_dim, map_dim, drop_rate):
        super(DistributionMLP, self).__init__()
        self.mapper = nn.Sequential(
            nn.Linear(source_dim, map_dim),
            nn.Dropout(drop_rate),
            #nn.LeakyReLU(),
            #nn.Linear(hid_dim, map_dim, bias=True),
            #nn.Dropout(drop_rate)
        )
        #self.linear = nn.Linear(source_dim, map_dim, bias=True)
    
    def forward(self, embedding, to_distribution=False):
        if to_distribution:
            map_emb = F.softmax(self.mapper(embedding), dim=1) #映射为概率分布
        else:
            map_emb = self.mapper(embedding)
        return map_emb


# 将预训练任务与下游任务的embedding映射到同一对比空间
class DenoiseMLP(nn.Module):
    def __init__(self, source_dim, hid_dim, map_dim, drop_rate):
        super(DenoiseMLP, self).__init__()
        self.mapper = nn.Sequential(
            nn.Linear(source_dim, hid_dim),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, map_dim),
            nn.Dropout(drop_rate)
        )
        #self.linear = nn.Linear(source_dim, map_dim, bias=True)
    
    def forward(self, embedding):
        map_emb = self.mapper(embedding)
        return map_emb
