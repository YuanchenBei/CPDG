import torch
import torch.nn as nn
import numpy as np

class AttTransformer(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super(AttTransformer, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.LeakyReLU(),
            nn.Linear(self.emb_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        candidates = x[:, :-1]
        num_candidates = candidates.shape[1]
        target = x[:, [-1] * num_candidates]
        #print("candidate shape: ", candidates.shape)
        #print("target shape: ", target.shape)
        feat = torch.cat([candidates, target], dim=-1)
        #print("feat shape: ", feat.shape)
        att = self.mlp(feat)
        weighted_behaviors = x[:, :-1].mul(att)
        evolutionary = weighted_behaviors.sum(dim=1)
        return evolutionary


class MeanOperator(nn.Module):
    def __init__(self):
        super(MeanOperator, self).__init__()

    def forward(self, pretrain_emb):
        return torch.mean(pretrain_emb, dim=1)


class EvolutionInfoGetter(nn.Module):
    def __init__(self, in_shape, out_shape, seq_len, mode=None):
        super(EvolutionInfoGetter, self).__init__()
        self.mode = mode
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.seq_len = seq_len
        # 三种模式：rnn, transformer, mean pooling
        if self.mode == 'rnn':
            self.getter = nn.GRU(self.in_shape, self.out_shape, batch_first=True)
        elif self.mode == 'lstm':
            self.getter = nn.LSTM(self.in_shape, self.out_shape, batch_first=True)
        elif self.mode == 'trans':
            self.getter = AttTransformer(self.seq_len, self.in_shape)
        else:
            self.getter = MeanOperator()

    def forward(self, pretrain_emb):
        if self.mode == 'rnn':
            _, evolution_info = self.getter(pretrain_emb)
        elif self.mode == 'lstm':
            _, (evolution_info, _) = self.getter(pretrain_emb)
        else:
            evolution_info = self.getter(pretrain_emb)
        return evolution_info


class AdaptiveFusion(nn.Module):
    def __init__(self, info_dim1, info_dim2, out_dim):
        super(AdaptiveFusion, self).__init__()
        self.info_dim1 = info_dim1
        self.info_dim2 = info_dim2
        self.out_dim = out_dim
        self.fusion_func = nn.Sequential(
            nn.Linear(self.info_dim1+self.info_dim2, self.out_dim),
            nn.LeakyReLU(),
            nn.Linear(self.out_dim, self.out_dim)
        )

    def forward(self, evolution_info, current_info):
        fusion_info = self.fusion_func(torch.cat([evolution_info, current_info], dim=1))
        return fusion_info
