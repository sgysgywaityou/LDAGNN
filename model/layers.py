import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, is_adaptive_weight=False, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.is_adaptive_weight = is_adaptive_weight

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))) # W.shape (in_features, out_features)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if not is_adaptive_weight:
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))) # a.shape (2*out_features, 1)
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, adaptive_weight=None):
        Wh = torch.mm(h, self.W)
        if adaptive_weight == None:
            e = self._prepare_attentional_mechanism_input(Wh) # (N, N)
            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            zero_vec = -9e15*torch.ones_like(adaptive_weight)
            attention = torch.where(adj > 0, adaptive_weight, zero_vec)
        attention = F.softmax(attention, dim=1) # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention, Wh = attention.to(torch.float32), Wh.to(torch.float32)
        h_prime = torch.matmul(attention, Wh) # (N, out_features)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(0, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolutionLayer(nn.Module):
    """
    L = E·H·W
    """
    def __init__(self, out_features, embed_dim=768, act=True) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(embed_dim, out_features))) # W.shape (embed_dim, out_features)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.act = act
    
    def forward(self, e, h):
        wh = torch.matmul(h, self.W)
        e, wh = e.to(torch.float32), wh.to(torch.float32)
        if self.act:
            return F.relu(torch.matmul(e, wh))
        else:
            return torch.matmul(e, wh)


class AdaptiveFusionLayer(nn.Module):
    def __init__(self, out_features, embed_dim=768) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(size=(embed_dim, out_features))) # W.shape (embed_dim, out_features)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))) # a.shape (2*out_features, 1)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.out_features = out_features

    def forward(self, F_0: torch.Tensor, F_1: torch.Tensor, F_2: torch.Tensor, F_3: torch.Tensor):
        N = len(F_0)
        F_con = torch.concat([F_1, F_2, F_3], dim=0) # (3*N, embed_dim)
        WF_0 = torch.matmul(F_0, self.W) # (N, out_features)
        WF_con = torch.matmul(F_con, self.W) # (3*N, out_features)
        e = self._prepare_attentional_mechanism_input(WF_0, WF_con)
        e = F.softmax(e, dim=1) # (N, 3)
        R = torch.empty(size=(N, self.out_features)) # (N, out_features)
        for idx in range(N):
            wh = torch.matmul(e[idx, :], torch.concat([F_1[idx].reshape(1, -1), F_2[idx].reshape(1, -1), F_3[idx].reshape(1, -1)], dim=0)) # (1, embed_dim)
            R[idx, :] = F.relu(torch.matmul(wh, self.W)) + F_0[idx]
        return R

    def _prepare_attentional_mechanism_input(self, WF_0, WF_con):
        Wh1 = torch.matmul(WF_0, self.a[:self.out_features, :]) # (N, 1)
        Wh2 = torch.matmul(WF_con, self.a[self.out_features:, :]) # (3*N, 1)
        Wh2 = Wh2.reshape(-1, 3)    # (N, 3)
        e = Wh1 + Wh2 # (N, 3)
        return e


class Dense(nn.Module):
    def __init__(self, in_features, label_nums) -> None:
        super().__init__()
        self.label_nums = label_nums
        self.linear = nn.Linear(in_features, label_nums)


    def forward(self, fv: torch.Tensor):
        return self.linear(fv)