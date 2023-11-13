import string
import re
from zhon.hanzi import punctuation
from typing import List
import jieba
import torch


def clean_text(text):

    punctuation_en = string.punctuation # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punctuation_zh = punctuation   # '＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。
    new_str = re.sub('[{}]'.format(punctuation_en), "", text)
    new_str = re.sub('[{}]'.format(punctuation_zh), "", new_str)
    new_str = re.sub('[a-zA-Z]', '', new_str)
    new_str = re.sub('[\d]', '', new_str)
    new_str = re.sub('[\s]', '', new_str)
    return new_str

def segmentation(document_path, stop_words) -> List[str]:
    document = ""
    with open(document_path, "r", encoding="utf-8") as f:
        lines:List[str] = f.readlines()
        document += "".join([line.strip() for line in lines])
        document = clean_text(document)
        words = jieba.lcut(document)
        words = [w for w in words if w not in stop_words and w.strip() != ""]
    return words

def get_word_vector(tokenizer, model, word):
    tokens = tokenizer(word, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    word_vector = outputs.last_hidden_state[0][0]
    return word_vector