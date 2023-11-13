#coding=utf-8
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_embedding_vector(text, max_len, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    tokenizer = BertTokenizer.from_pretrained(
        r".\bert_base_uncased_pytorch")
    model = BertModel.from_pretrained(
        r".\bert_base_uncased_pytorch")
    model = model.to(device)
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)
    with torch.no_grad():
        last_hidden_states = model(tokens_tensor)[0]

    token_embeddings = []
    if max_len==None:
        max_len = len(tokenized_text)
    if len(tokenized_text)>=max_len:
        print('1')
        for token_i in range(max_len):
            hidden_layers = []
            for layer_i in range(len(last_hidden_states)):
                vec = last_hidden_states[layer_i][0][token_i]
                vec = vec.to('cpu')
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        summed_last_4_layers = [torch.sum(torch.stack(
            layer)[-4:], 0) for layer in token_embeddings]
        output = torch.stack(summed_last_4_layers, dim=0)
        return output.to('cpu')
    else:
        print('2')
        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            for layer_i in range(len(last_hidden_states)):
                vec = last_hidden_states[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        summed_last_4_layers = [torch.sum(torch.stack(
            layer)[-4:], 0) for layer in token_embeddings]

        for plus_i in range(max_len-len(tokenized_text)):
            summed_last_4_layers.append(torch.zeros(768).to(device))

        output = torch.stack(summed_last_4_layers, dim=0)
        return output.to('cpu')