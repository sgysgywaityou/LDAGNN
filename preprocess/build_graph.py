from typing import List
import os
import numpy as np
from scipy import sparse
import pickle
from typing import Dict, Set
import math
import yaml
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from utils import get_word_vector


def build_G(root_path, data_path, bert_path, word2idx_file, window_size=20, embed_dim=768):
    with open(word2idx_file, "rb") as f:
        word2idx: Dict[str, int] = pickle.load(f)
    word_idx_pairs = sorted(word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab: List[str] = [w[0] for w in word_idx_pairs]

    documents_list: List[str] = []
    with open(file=data_path, mode='r', encoding='utf-8') as f:
        documents_list = f.readlines()
    document_nums = len(documents_list)
    vocab_nums = len(vocab)
    print(f"total words: {vocab_nums}, total documents: {document_nums}")
    
    windows: List[List[str]] = []
    for document in documents_list:
        document_words: List[str] = document.split()
        length = len(document_words)
        if length <= window_size:
            windows.append(document_words)
        else:
            for j in range(length - window_size + 1):
                window = document_words[j: j + window_size]
                windows.append(window)
    print("total windows: ", len(windows))

    word_window_freq: Dict[str, int] = {}
    for window in windows:
        appeared: Set[str] = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    
    word_pair_count: Dict[str, int] = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word2idx[word_i]
                word_j = window[j]
                word_j_id = word2idx[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    print("Finish word pairs statistic.")

    row = []
    col = []
    weight = []

    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = math.log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(document_nums + i)
        col.append(document_nums + j)
        weight.append(pmi)
    print("Finish constructing edges between word node and word node done(PMI)!")

    word_doc_list: Dict[str, List[int]] = {}

    for i in range(len(documents_list)):
        doc_words = documents_list[i]
        words: List[str] = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    doc_word_freq: List[str, int] = {}

    for doc_id in range(len(documents_list)):
        doc_words: str = documents_list[doc_id]
        words: List[str] = doc_words.split()
        for word in words:
            word_id = word2idx[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(documents_list)):
        doc_words = documents_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word2idx[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            row.append(i)
            col.append(document_nums + j)
            idf = math.log(1.0 * len(documents_list) / word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            row.append(document_nums + j)
            col.append(i)
            weight.append(freq * idf)
            doc_word_set.add(word)
    print("Finish constructing edges between word node and document node done(TF-IDF)!")
    node_nums = document_nums + vocab_nums
    for i in range(node_nums):
        row.append(i)
        col.append(i)
        weight.append(1)
    
    print("saving initial graph...")
    adj_g = sparse.coo_matrix((weight, (row, col)), shape=(node_nums, node_nums))
    with open(f"{root_path}/w_d.train.adj", 'wb') as f:
        pickle.dump(adj_g, f)
    print("Finished constructing graph A including word nodes and document nodes!")
    degree_row = []
    degree_col = []
    degree_weight = []
    print("saving degree graph...")
    for idx in range(node_nums):
        print(f"coping node {idx}/{node_nums}")
        degree_row.append(idx)
        degree_col.append(idx)
        degree_weight.append(sum(adj_g.getrow(idx).toarray().reshape(-1)))
    degree_adj = sparse.coo_matrix((degree_weight, (degree_row, degree_col)), shape=(node_nums, node_nums))
    with open(f"{root_path}/degree.train.adj", 'wb') as f:
        pickle.dump(degree_adj, f)
    print("Finished constructing degree graph of A including word nodes and document nodes!")

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path)
    labels: List[str] = os.listdir("../data/THUCNews/train/")
    label2vec: Dict[str, torch.Tensor] = {}
    label2idx: Dict[str, int] = {}
    label_vec_matrix = []
    word2vec: Dict[str, torch.Tensor] = {}
    for idx in range(len(labels)):
        label = labels[idx]
        label2vec[label] = get_word_vector(tokenizer, model, label)
        label_vec_matrix.append(label2vec[label])
        label2idx[label] = idx

    label_vec_matrix = torch.stack(label_vec_matrix, dim=0)    # (label_nums, embed_dim)
    with open(f"{root_path}/label_vec_matrix.pkl", "wb") as f:
        pickle.dump(label_vec_matrix, f)

    w_label_row = []
    w_label_col = []
    w_label_weight = []
    for word_idx in range(len(vocab)):
        word = vocab[word_idx]
        word_vec = get_word_vector(tokenizer, model, word)
        for label_idx in range(len(labels)):
            label = labels[label_idx]
            label_vec = label2vec[label]
            cos_sim = F.cosine_similarity(word_vec, label_vec, dim=0)
            w_label_row.append(document_nums+word_idx)
            w_label_col.append(node_nums+label_idx)
            w_label_weight.append(cos_sim)
            w_label_row.append(node_nums+label_idx)
            w_label_col.append(document_nums+word_idx)
            w_label_weight.append(cos_sim)
        if word not in word2vec:
            word2vec[word] = word_vec
    print("construct edges between word node and label node done!")

    vocab_vec_matrix = [word2vec[vocab[index]].reshape(1, -1) for index in range(vocab_nums)]
    vocab_vec_matrix = torch.concat(vocab_vec_matrix, dim=0) # (vocab_nums, embed_dim)
    with open(f"{root_path}/word_vec_matrix.pkl", "wb") as f:
        pickle.dump(vocab_vec_matrix, f)

    doc_vec_list: List[torch.Tensor] = []
    d_label_row = []
    d_label_col = []
    d_label_weight = []
    for doc_idx in range(len(documents_list)):
        document: str = documents_list[doc_idx]
        document_words: List[str] = document.split()
        all_words_vec: List[torch.Tensor] = [word2vec[w] for w in document_words]
        doc_len = len(document_words)

        doc_vec: torch.Tensor = sum(all_words_vec) / doc_len
        doc_vec_list.append(doc_vec)
        for label_idx in range(len(labels)):
            label = labels[label_idx]
            label_vec = label2vec[label]
            att = torch.dot(doc_vec, label_vec)
            d_label_row.append(doc_idx)
            d_label_col.append(node_nums+label_idx)
            d_label_weight.append(att)
            d_label_row.append(node_nums+label_idx)
            d_label_col.append(doc_idx)
            d_label_weight.append(att)

    doc_vec_list = torch.stack(doc_vec_list, dim=0) # => (doc_nums, embed_dim)
    with open(f"{root_path}/doc_vec_matrix.pkl", "wb") as f:
        pickle.dump(doc_vec_list, f)
    print("construct edges between document node and label node done!")
    
    all_node_nums = node_nums + len(labels)
    print("saving graph 1...")
    w_label_adj = sparse.coo_matrix((weight+w_label_weight, (row+w_label_row, col+w_label_col)), shape=(all_node_nums, all_node_nums))
    with open(f"{root_path}/w_d_l_edge_w-l.train.adj", 'wb') as f:
        pickle.dump(w_label_adj, f)
    print("Finished constructing graph A_1 including word nodes, document nodes and label nodes(edges between word and label nodes)!")
    degree_row = []
    degree_col = []
    degree_weight = []
    print("saving degree graph 1...")
    for idx in range(all_node_nums):
        print(f"coping node {idx}/{all_node_nums}")
        degree_row.append(idx)
        degree_col.append(idx)
        degree_weight.append(sum(w_label_adj.getrow(idx).toarray().reshape(-1)))
    degree_adj = sparse.coo_matrix((degree_weight, (degree_row, degree_col)), shape=(all_node_nums, all_node_nums))
    with open(f"{root_path}/degree_edge_w-l.train.adj", 'wb') as f:
        pickle.dump(degree_adj, f)
    print("Finished constructing degree graph of A_1 including word nodes, document nodes and label nodes(edges between word nodes and label nodes)!")

    print("saving graph 2...")
    d_label_adj = sparse.coo_matrix((weight+d_label_weight, (row+d_label_row, col+d_label_col)), shape=(all_node_nums, all_node_nums))
    with open(f"{root_path}/w_d_l_edge_d-l.train.adj", 'wb') as f:
        pickle.dump(d_label_adj, f)
    print("Finished constructing graph A_2 including word nodes, document nodes and label nodes(edges between document and label nodes)!")
    degree_row = []
    degree_col = []
    degree_weight = []
    print("saving degree graph 2...")
    for idx in range(all_node_nums):
        print(f"coping node {idx}/{all_node_nums}")
        degree_row.append(idx)
        degree_col.append(idx)
        degree_weight.append(sum(d_label_adj.getrow(idx).toarray().reshape(-1)))
    degree_adj = sparse.coo_matrix((degree_weight, (degree_row, degree_col)), shape=(all_node_nums, all_node_nums))
    with open(f"{root_path}/degree_edge_d-l.train.adj", 'wb') as f:
        pickle.dump(degree_adj, f)
    print("Finished constructing degree graph of A_2 including word nodes, document nodes and label nodes(edges between document nodes and label nodes)!")

    print("saving graph 3...")
    w_d_label_adj = sparse.coo_matrix((weight+w_label_weight+d_label_weight, (row+w_label_row+d_label_row, col+w_label_col+d_label_col)), shape=(all_node_nums, all_node_nums))
    with open(f"{root_path}/w_d_l_edge_w-d-l.train.adj", 'wb') as f:
        pickle.dump(w_d_label_adj, f)
    print("Finished constructing graph A_3 including word nodes, document nodes and label nodes(edges between document and label nodes, word and label nodes)!")
    degree_row = []
    degree_col = []
    degree_weight = []
    print("saving degree graph 3...")
    for idx in range(all_node_nums):
        print(f"coping node {idx}/{all_node_nums}")
        degree_row.append(idx)
        degree_col.append(idx)
        degree_weight.append(sum(w_d_label_adj.getrow(idx).toarray().reshape(-1)))
    degree_adj = sparse.coo_matrix((degree_weight, (degree_row, degree_col)), shape=(all_node_nums, all_node_nums))
    with open(f"{root_path}/degree_edge_w-d-l.train.adj", 'wb') as f:
        pickle.dump(degree_adj, f)
    print("Finished constructing degree graph of A_3 including word nodes, document nodes and label nodes(edges between word nodes and label nodes, document nodes and label nodes)!")
    

if __name__ == "__main__":
    data_config_file = "../data/config.yaml"
    with open(data_config_file, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    data_path = config['corpus_path']
    train_data = data_path + config['train_data']['data_file']
    test_path = config['test_path']
    word2idx_file = config['corpus_path'] + config['word2idx_file']
    bert_path = config['../model/chinese_wwm_ext_pytorch/']
    window_size = config['window_size']
    build_G(root_path=data_path, data_path=train_data, bert_path=bert_path, word2idx_file=word2idx_file, window_size=window_size)