from typing import List
import os
import pickle
import yaml
from utils import segmentation


def clean_corpus(data_path:str, clean_path:str, word2idx_file:str, stop_words:List[str], thresh=3):
    word_frequency = {}
    doc_counter = 1
    word_counter = 0
    word2idx = {}
    for label in os.listdir(data_path):
        document_list = os.listdir(data_path+label)
        for document_name in document_list:
            document_words = segmentation(data_path+label+"/"+document_name, stop_words)
            for word in document_words:
                if word in word_frequency:
                    word_frequency[word] += 1
                else:
                    word_frequency[word] = 1
            print(f"statistic document {doc_counter} finished...")
            doc_counter += 1
    doc_counter = 1
    clean_docs = []
    label_docs = []
    label_names = os.listdir(data_path)
    for label_idx in range(len(label_names)):
        label = label_names[label_idx]
        document_list = os.listdir(data_path+label)
        for document_name in document_list:
            doc_words = []
            document_words = segmentation(data_path+label+"/"+document_name, stop_words)
            for word in document_words:
                if word_frequency[word] >= thresh:
                    doc_words.append(word)
                    if word not in word2idx:
                        word2idx[word] = word_counter
                        word_counter += 1
            clean_docs.append(' '.join(doc_words))
            print(f"document {doc_counter} clean finished...")
            doc_counter += 1
            label_docs.append(str(label_idx))
    clean_corpus_content = "\n".join(clean_docs)
    clean_corpus_labels = "\n".join(label_docs)
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    with open(file=clean_path+"train_corpus.txt", mode="w", encoding="utf-8", newline="") as f:
        f.write(clean_corpus_content)
    with open(file=clean_path+"train_label.txt", mode="w", encoding="utf-8", newline="") as f:
        f.write(clean_corpus_labels)
    print("clean corpus write finished!")
    with open(clean_path+word2idx_file, "wb") as f:
        pickle.dump(word2idx, f)
    print("word2index saved...total words: ", word_counter)

data_config_file = "../data/config.yaml"
with open(data_config_file, 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

stop_path = config['../data/hit_stopwords.txt']
stop_words = []
with open(stop_path, "r", encoding="utf-8") as f:
    lines:List[str] = f.readlines()
    lines = [line.strip() for line in lines]
    stop_words += lines

train_path = config['train_path']
test_path = config['test_path']
word2idx_file = config['word2idx_file']
clean_corpus_path = config['corpus_path']
clean_corpus(data_path=train_path, clean_path=clean_corpus_path, word2idx_file=word2idx_file, stop_words=stop_words)

