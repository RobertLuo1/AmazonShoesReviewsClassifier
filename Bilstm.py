#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import torch
from torch import nn
import torch.nn.functional as F
import config
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

class BiRNN(nn.Module):
    """
    The neural network 
    """
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,device,padding_idx):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size,padding_idx=padding_idx)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.fc1 = nn.Linear(2*num_hiddens, 200)
        # self.fc1 = nn.Linear(num_hiddens, 200)
        self.fc2 = nn.Linear(200,50)
        self.fc3 = nn.Linear(50,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchNornalize1 = nn.BatchNorm1d(200)
        self.batchNornalize2 = nn.BatchNorm1d(50)
        self.w_omega = nn.Parameter(torch.Tensor(num_hiddens*2,num_hiddens*2))
        # self.w_omega = nn.Parameter(torch.Tensor(num_hiddens,num_hiddens))
        self.u_omege = nn.Parameter(torch.Tensor(num_hiddens*2,1))
        # self.u_omege = nn.Parameter(torch.Tensor(num_hiddens,1))
        # self.multihead = nn.MultiheadAttention(embed_size,num_heads=8,batch_first=True,dropout=True)
        nn.init.uniform_(self.w_omega,-0.1,0.1)
        nn.init.uniform_(self.u_omege,-0.1,0.1)
        for name, params in self.encoder.named_parameters():
            # weight: Orthogonal Initialization
            if 'weight' in name:
                nn.init.orthogonal_(params)
            # lstm forget gate bias init with 1.0
            if 'bias' in name:
                b_i, b_f, b_c, b_o = params.chunk(4, 0)
                # nn.init.ones_(b_f)
                nn.init.uniform_(b_f,0,1)
        self.to(device)

    def forward(self, inputs,input_length):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs)
        embeddings = pack_padded_sequence(embeddings, input_length, batch_first=True)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        outputs, (h_n,c_n) = self.encoder(embeddings) # output, (h, c)
        outputs,_ = pad_packed_sequence(outputs,batch_first=True)
        # output_fw = h_n[-2, :, :]  # 正向最后一次输出
        # output_bw = h_n[-1, :, :]  # 反向最后一次输出
        # output = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size,2*hidden_size]
        u = torch.tanh(torch.matmul(outputs,self.w_omega))
        att = torch.matmul(u,self.u_omege)
        att_score = F.softmax(att,dim=1)
        score_x = outputs*att_score
        output = torch.sum(score_x,dim=1)
        #Attention 机制
        output = self.fc1(output)
        output = self.relu(output)
        output = self.batchNornalize1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.batchNornalize2(output)
        output = self.dropout(output)
        output = self.fc3(output)
        # output = self.sigmoid(output)
        return output