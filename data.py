import re
import gensim
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写'''
def get_word_list(query):
    regEx = re.compile('[\\W]+')#我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')#[\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)#切分文本
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

# 加载数据
def load_sentences(file, data_size=None):
    df = pd.read_csv(file)
    p = map(get_word_list, df['sentence1'].values[0:data_size])#读取并且分句子1
    h = map(get_word_list, df['sentence2'].values[0:data_size])#读取并切分句子2
    label = df['label'].values[0:data_size]#读取标签
    return p, h, label

# 加载字典
def load_vocab(vocab_file):
    #读取vocab文件
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
    #构造word转id，和id转word的字典
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab


def load_embeddings(embdding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    #填充向量矩阵
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]#词向量矩阵
    return embedding_matrix

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    lengths = [len(s) for s in sequences]#计算所有句子长度
    nb_samples = len(sequences)#计算出最长的句子长度
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)#初始化所有句子
    for idx, s in enumerate(sequences):#遍历每一个句子
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':#截断
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':#填充
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

# word->index
def word_index(p_sentences, h_sentences, word2idx, max_char_len):
    p_list, p_length, h_list, h_length = [], [], [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):#遍历所有句子
        p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]#句子1转id
        h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]#句子2转id
        p_list.append(p)
        p_length.append(min(len(p), max_char_len))#保存真实长度
        h_list.append(h)
        h_length.append(min(len(h), max_char_len))
    p_list = pad_sequences(p_list, maxlen = max_char_len)#填充
    h_list = pad_sequences(h_list, maxlen = max_char_len)
    return p_list, p_length, h_list, h_length


class MyDataset(Dataset):
    def __init__(self, file, vocab_file, max_char_len):
        p, h, self.label = load_sentences(file)  # 加载数据
        word2idx, _, _ = load_vocab(vocab_file)  # 加载词表
        # 转id并填充
        self.p_list, self.p_lengths, self.h_list, self.h_lengths = word_index(p, h, word2idx, max_char_len)
        self.p_list = torch.from_numpy(self.p_list).type(torch.long)  # 转成torch的Tensor
        self.h_list = torch.from_numpy(self.h_list).type(torch.long)
        self.max_length = max_char_len

    def __len__(self):  # 重写len方法
        return len(self.label)

    def __getitem__(self, idx):  # 重写方法
        return self.p_list[idx], self.p_lengths[idx], self.h_list[idx], self.h_lengths[idx], self.label[idx]
