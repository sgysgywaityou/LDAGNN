import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loader import DocData, load_w_d_label_sp, load_vecs, load_degree, load_sub_adj
from typing import List, Dict
import os
import yaml
from model.models import LDAGNN
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model_config_file = "../model/config.yaml"
    with open(model_config_file, 'r') as f:
        model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset_root = "./data/THUCNews/train/"
    root_path = "./data/THUCNews/corpus/"
    data_file = "train_corpus.txt"
    label_file = "train_label.txt"
    word2idx_file = "word_index.pkl"
    degree_file = "degree.train.adj"
    degree_w_label_file = "degree_edge_w-l.train.adj"
    degree_d_label_file = "degree_edge_d-l.train.adj"
    degree_w_d_label_file = "degree_edge_w-d-l.train.adj"
    word_vecs_file = "word_vec_matrix.pkl"
    doc_vecs_file = "doc_vec_matrix.pkl"
    label_vecs_file = "label_vec_matrix.pkl"
    train_per_class = 5
    ratio = 0.8
    labels = os.listdir(dataset_root)
    embed_dim = model_config['embed_dim']
    dropout = model_config['dropout']
    alpha = model_config['alpha']
    delta = model_config['delta']
    beta = model_config['beta']
    eta = model_config['eta']
    mu = model_config['mu']
    k = model_config['nhead']

    degree_matrix, degree_w_label, degree_d_label, degree_w_d_label = load_degree(degree_path=root_path+degree_file), load_degree(
        degree_path=root_path+degree_w_d_label_file), load_degree(degree_path=root_path+degree_d_label_file), load_degree(degree_path=root_path+degree_w_d_label_file)
    word_doc_sp, doc_label_sp, word_label_sp, d_w_label_sp = load_w_d_label_sp(
        root=root_path)
    word_doc_adj, doc_label_adj, word_label_adj, d_w_label_adj = torch.tensor(word_doc_sp.toarray(
    )), torch.tensor(doc_label_sp.toarray()), torch.tensor(word_label_sp.toarray()), torch.tensor(d_w_label_sp.toarray())
    word_vec_matrix, doc_vec_matrix, label_vec_matrix = load_vecs(vecs_path=root_path+word_vecs_file), load_vecs(
        vecs_path=root_path+doc_vecs_file), load_vecs(vecs_path=root_path+label_vecs_file)

    word_nums = len(word_vec_matrix)
    doc_nums = int(train_per_class*len(labels)*ratio)
    print(f"total word nodes: ", word_nums)
    print(f"total document nodes: ", doc_nums)
    print(f"total label nodes: ", len(labels))
    print("Finish loading sparse matrix successfully!")

    batch_size = 16
    epochs = 10
    class_nums = len(labels)
    label_onehots = torch.eye(class_nums)
    doc_dataset = DocData(root_path=root_path, label_file=label_file,
                          data_file=data_file, word2idx_file=word2idx_file)
    train_loader = DataLoader(
        dataset=doc_dataset, batch_size=batch_size, shuffle=True)
    model = LDAGNN(in_features=embed_dim, out_features=embed_dim, nclass=class_nums, dropout=dropout,
                   alpha=alpha, delta=delta, beta=beta, eta=eta, mu=mu, nheads=k, embed_dim=embed_dim, concat=True)
    loss_f = nn.CrossEntropyLoss()

    train_loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        true_counter = 0
        for idx, (doc_words_idxs, labels, lengths, doc_idxs) in enumerate(train_loader):
            label_true = label_onehots[labels]  # (b, nclass)
            appeared_words = set()
            appeared_labels_list: List[int] = list(set(labels))
            for doc_words_idx in doc_words_idxs:
                appeared_words.update(doc_words_idx.tolist())
            if -1 in appeared_words:
                appeared_words.remove(-1)  # 去除padding
            appeared_words_list: List[int] = list(
                appeared_words)
            doc_wordidx_list: List = []
            for doc_words_idx, doc_idx, doc_len in zip(doc_words_idxs.tolist(), doc_idxs.tolist(), lengths.tolist()):
                word_idx_in_appeared = [appeared_words_list.index(
                    word_idx) for word_idx in doc_words_idx[:doc_len]]
                doc_wordidx_list.append(word_idx_in_appeared)
            w_d_idxs: List[int] = doc_idxs.tolist().copy(
            ) + (torch.tensor(appeared_words_list)+doc_nums).tolist().copy()
            w_d_label_idxs: List[int] = w_d_idxs.copy() + (torch.tensor(appeared_labels_list) +
                                                word_nums + doc_nums).tolist()

            sub_w_d_adj = word_doc_adj[w_d_idxs][:, w_d_idxs]
            sub_doc_label_adj = doc_label_adj[w_d_label_idxs][:, w_d_label_idxs]
            sub_word_label_adj = word_label_adj[w_d_label_idxs][:, w_d_label_idxs]
            sub_d_w_label_adj = d_w_label_adj[w_d_label_idxs][:, w_d_label_idxs]

            sub_w_d_degree = load_sub_adj(degree_matrix, w_d_idxs)  # (N, N)
            sub_w_label_degree = load_sub_adj(degree_w_label, w_d_label_idxs) # (N+L, N+L)
            sub_d_label_degree = load_sub_adj(degree_d_label, w_d_label_idxs) # (N+L, N+L)
            sub_w_d_label_degree = load_sub_adj(degree_w_d_label, w_d_label_idxs) # (N+L, N+L)
            sub_word_vecs = word_vec_matrix[appeared_words_list]
            sub_doc_vecs = doc_vec_matrix[doc_idxs]
            sub_label_vecs = label_vec_matrix[labels]

            sub_node_vecs = torch.concat(
                [sub_doc_vecs, sub_word_vecs], dim=0)  # (N, embed_dim)
            sub_all_node_vecs = torch.concat(
                [sub_doc_vecs, sub_word_vecs, sub_label_vecs], dim=0)  # (N+L, embed_dim)
            out: torch.Tensor = model(sub_node_vecs, sub_all_node_vecs, sub_w_d_adj, sub_word_label_adj, sub_doc_label_adj,
                        sub_d_w_label_adj, sub_w_d_degree, sub_w_d_label_degree, doc_wordidx_list, len(sub_doc_vecs))  # (b, nclass)
            true_counter_epoch = (labels == torch.argmax(out, dim=1)).sum()
            true_counter += true_counter_epoch
            loss = loss_f(out, labels).detach().cpu()
            epoch_loss += loss
            print(f"epoch {epoch+1}, batch {idx}, loss: {loss}, accuracy: {true_counter_epoch/batch_size}")
        train_loss_list.append(epoch_loss)
        print(f"epoch {epoch+1} over... total loss: {epoch_loss}, total accuracy: {true_counter/doc_nums}")
        
    plt.figure()
    plt.plot(train_loss_list)
    plt.legend()
    plt.title("Loss curve")
    plt.show()
