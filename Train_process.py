#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import torch
from textdataset import getdataLoader
import config
from Bilstm import BiRNN
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR
import random
import torch.nn as nn


def train(train_dataloader, model, device, optimizer, criterion,scheduler):
    train_loss = 0
    Data_train = train_dataloader
    model.train()
    for i, (text, label,input_length) in enumerate(Data_train):
        optimizer.zero_grad()
        text, label = text.to(device), label.to(device)
        output = model(text,input_length)
        loss = criterion(output, label)
        # print(loss.item())
        train_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=20,norm_type=2)
        optimizer.step()
        scheduler.step()
    return train_loss

def test(test_dataloader, model, device):
    # test_loss = 0
    accuracy_list = []
    model.eval()
    Data_test = test_dataloader
    for text, label,input_length in Data_test:
        text, label = text.to(device), label.to(device)
        with torch.no_grad():
            output = model(text,input_length)
            pred = output.max(dim=-1)[-1]  # 对最后一个维度取最大值，[batch,hidden_size] 对最后一个维度取最大值，
            cur_acc = pred.eq(label).float().mean()
            accuracy = cur_acc.cpu().item()
            accuracy_list.append(accuracy)
    accuracy_mean = np.array(accuracy_list).mean()
    return accuracy_mean


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def Train():
    # setup_seed(42)
    device = config.device
    train_dataloader = getdataLoader("train")
    test_dataloader = getdataLoader("test")

    N_EPOCHS = config.epoch
    padding_index = config.ws.PAD
    embed_size, num_hiddens, num_layers = config.embedding_size,config.num_layer,config.num_layer
    learning_rate = config.learning_rate
    model = BiRNN(config.ws, embed_size, num_hiddens, num_layers,device,padding_index)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = MultiStepLR(
        optimizer,
        # [10,20,30],
        [40],
        gamma = 0.1
    )
    # scheduler = ExponentialLR(optimizer,gamma=0.95)
    running_loss = []
    accuracy_list = []
    for epoch in range(N_EPOCHS):
        training_loss = train(train_dataloader, model, device, optimizer, criterion,scheduler)
        running_loss.append(training_loss)
        accuracy = test(test_dataloader, model, device)
        accuracy_list.append(accuracy)
        print(f'epoch : {epoch} ',f' training_loss : {training_loss}',f' accuracy : {accuracy}')
    plt.subplot(2,1,1)
    plt.plot(running_loss,label='loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(accuracy_list,label='accuray')
    plt.legend()
    plt.show()
    # plt.savefig('./picture/' + 'Bilstm_models' + str(N_EPOCHS) + ' ' + f'{running_loss[-1]:.2f}')
    # w_omega = model.state_dict()['w_omega']
    # print(w_omega.size())
    torch.save(model, './Bilstm_models' + str(N_EPOCHS)+' ' + f'{running_loss[-1]:.2f}'+'.pth')


if __name__ == '__main__':
    Train()



