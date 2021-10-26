import torch
import torch.nn as nn
import torch.nn.functional as F


class SiaGRU(nn.Module):
    def __init__(self, embeddings, hidden_size=300,max_len=50, num_layer=2,drop_out=0.2, device="gpu"):
        super(SiaGRU, self).__init__()
        self.device = device#设备
        self.drop_out=drop_out
        self.embeds_dim = embeddings.shape[1]#字嵌入维度
        self.word_emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1])#嵌入层
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings))#加载初始权重值
        self.word_emb.float()
        self.word_emb.weight.requires_grad = True#嵌入层参与训练
        self.word_emb.to(device)
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        #构建Bilstm层
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))#初始化状态向量
        self.h0.to(device)
        self.pred_fc = nn.Linear(max_len, 2)#全连接层  映射分类


    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward(self, sent1, sent2):
        p_encode = self.word_emb(sent1)  # 对句子1进行字嵌入
        h_encode = self.word_emb(sent2)  # 对句子2进行字嵌入
        # 随机失活
        p_encode = F.dropout(p_encode, p=self.drop_out, training=self.training)
        h_encode = F.dropout(h_encode, p=self.drop_out, training=self.training)

        encoding1, _ = self.gru(p_encode)  # 对句子1进行表示编码
        encoding2, _ = self.gru(h_encode)  # 对句子2进行表示编码
        # 计算指数相似向量
        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        # 对指数相似向量进行全连接映射
        x = self.pred_fc(sim.squeeze(dim=-1))
        # 计算二分类softmax输出值
        probabilities = nn.functional.softmax(x, dim=-1)
        return x, probabilities