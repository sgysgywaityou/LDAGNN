from torch.utils.data import Dataset
import torch
from typing import List, Tuple, Dict
import pickle
from scipy import sparse


def load_vecs(vecs_path):
    with open(vecs_path, "rb") as f:
        return pickle.load(f)

def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        return pickle.load(f)

def load_word2idx(word2idx_path):
    with open(word2idx_path, "rb") as f:
        return pickle.load(f)

def load_docs(docs_path):
    with open(docs_path, "r", encoding="utf-8") as f:
        return f.readlines()

def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return f.readlines()

def load_degree(degree_path) -> sparse._coo.coo_matrix:
    with open(degree_path, "rb") as f:
        return pickle.load(f)

def load_w_d_label_sp(root) -> Tuple[sparse._coo.coo_matrix, sparse._coo.coo_matrix, sparse._coo.coo_matrix, sparse._coo.coo_matrix]:
    with open(root+"/w_d_l_edge_d-l.train.adj", "rb") as f:
        doc_label_sp = pickle.load(f)
    with open(root+"/w_d_l_edge_w-l.train.adj", "rb") as f:
        word_label_sp = pickle.load(f)
    with open(root+"/w_d_l_edge_w-d-l.train.adj", "rb") as f:
        d_w_label_sp = pickle.load(f)
    with open(root+"/w_d.train.adj", "rb") as f:
        word_doc_sp = pickle.load(f)
    return word_doc_sp, doc_label_sp, word_label_sp, d_w_label_sp


def load_sub_adj(sp_matrix: sparse._coo.coo_matrix, node_idxs: List[int]) -> torch.Tensor:
    sub_tensors = []
    for idx in node_idxs:
        t = torch.tensor(sp_matrix.getrow(idx).toarray())
        sub_tensors.append(t)
    sub_tensors = torch.concat(sub_tensors, dim=0)  # (n, N)    n=len(node_idxs)
    return sub_tensors[:, node_idxs] # (n, n)


class DocData(Dataset):
    def __init__(self, root_path, data_file, label_file, word2idx_file, len_thresh=300) -> None:
        super().__init__()
        self.root = root_path
        self.data_file = data_file
        self.label_file = label_file
        self.thresh = len_thresh
        self.doc_list: List[str] = load_docs(root_path+data_file)
        self.label_list: List[str] = load_labels(root_path+label_file)
        self.word2idx: Dict[str, int] = load_word2idx(root_path+word2idx_file)
     
    def __getitem__(self, index):
        doc_words: List[str] = self.doc_list[index].split()
        doc_len = len(doc_words)
        doc_word_idxs: List[int] = [self.word2idx[w] for w in doc_words]
        if doc_len < self.thresh:
            doc_word_idxs += [-1] * (self.thresh - doc_len)
        else:
            doc_word_idxs = doc_word_idxs[:self.thresh]
        return torch.tensor(doc_word_idxs), int(self.label_list[index]), doc_len, index

    def __len__(self):
        return len(self.doc_list)