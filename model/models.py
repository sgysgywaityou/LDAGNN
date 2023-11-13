import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import GraphAttentionLayer, GraphConvolutionLayer, AdaptiveFusionLayer, Dense


def positional_encoding(seq_len, d_model) -> torch.Tensor:
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float32)

class GraphMultiAttention(nn.Module):
    def __init__(self, in_features, hid_features, dropout, alpha, nheads, is_adaptive_weight=False, embed_dim=768):
        """Dense version of GAT."""
        super(GraphMultiAttention, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_features, hid_features, dropout=dropout, alpha=alpha, is_adaptive_weight=is_adaptive_weight, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hid_features * nheads, embed_dim, dropout=dropout, alpha=alpha, is_adaptive_weight=is_adaptive_weight, concat=False)

    def forward(self, x, adj, adaptive_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, adaptive_weight)[0] for att in self.attentions], dim=1) # => (N, hid_features*k)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, adaptive_weight)[0]) # => (N, hid_features)
        return x # => (N, hid_features)


class LDAGNN(nn.Module):
    def __init__(self, in_features, out_features, nclass, dropout, alpha, delta, beta, eta, mu, doc_len_thresh=200, nheads=4, embed_dim=768, concat=True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.doc_thresh = doc_len_thresh
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.mu = mu
        self.att_layer = GraphAttentionLayer(
            in_features, out_features, dropout, alpha, False, concat)
        self.multi_atten = GraphMultiAttention(in_features, embed_dim, dropout, alpha, nheads, True, embed_dim)
        self.gcn = GraphConvolutionLayer(
            out_features, embed_dim=embed_dim, act=True)
        self.fusion_layer = AdaptiveFusionLayer(out_features, embed_dim)
        self.classifier = Dense(doc_len_thresh*embed_dim, nclass)

    def forward(self, H, H_, A, A_1, A_2, A_3, D, D_, doc_words_idxs, sub_doc_len):
        F_0 = self.branch_forward(H, A, D)
        F_1 = self.branch_forward(H_, A_1, D_)
        F_2 = self.branch_forward(H_, A_2, D_)
        F_3 = self.branch_forward(H_, A_3, D_)
        w_d_node_nums = len(F_0)

        R = self.fusion_layer(F_0, F_1[:w_d_node_nums], F_2[:w_d_node_nums], F_3[:w_d_node_nums]) # (N, out_features=embed_dim)
        b = len(doc_words_idxs)
        FV = []
        for doc_words_idx in doc_words_idxs:
            doc_words_idx = (torch.tensor(doc_words_idx) + sub_doc_len).tolist()
            doc_nv = R[doc_words_idx, :]   # (doc_len, embed_dim)
            doc_len = len(doc_words_idx)
            pos_enc = positional_encoding(doc_len, self.embed_dim) # (doc_len, embed_dim)
            doc_fv = doc_nv + pos_enc # (doc_len, embed_dim) ===> (doc_thresh, embed_dim)
            if doc_len < self.doc_thresh:
                doc_fv = torch.concat([doc_fv, torch.zeros(self.doc_thresh-doc_len, self.embed_dim)], dim=0)
            else:
                doc_fv = doc_fv[:self.doc_thresh]
            print(doc_fv)
            FV.append(doc_fv.reshape(1, self.doc_thresh, self.embed_dim))
        FV = torch.concat(FV, dim=0)    # (batch, doc_thresh, embed_dim)
        out = self.classifier(FV.reshape(b, -1))
        return F.softmax(F.tanh(out), dim=1)

    def branch_forward(self, H, A_i, D):
        O = self.att_layer(H, A_i)[1] # O: (N, N)
        E = self.delta * A_i + self.beta * O
        L = self.multi_atten(H, A_i, E)
        D_diag = torch.diag(D)
        D_diag = torch.pow(D_diag, -0.5)
        D_diag[torch.isinf(D_diag)] = 0
        D_diag = torch.diag(D_diag)
        E_ = torch.matmul(torch.matmul(D_diag, E), D_diag) # (N, N)·(N, N)·(N, N)
        Z = self.gcn(E_, H)
        F_i = self.eta * L + self.mu * Z
        return F_i