#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import torch
from torch.utils.data import DataLoader,Dataset
from config import ws,max_len,train_batch_size,test_batch_size
import os
from nltk import word_tokenize
import numpy as np

class textDataset(Dataset):
    def __init__(self,mode):
        super(textDataset,self).__init__()
        data_base_file = r'../data/dataset'
        if mode =='train':
            review_path = os.path.join(data_base_file,'train')
        else:
            review_path = os.path.join(data_base_file,'test')
        temp_data_path = [os.path.join(review_path, 'pos'), os.path.join(review_path, 'neg')]
        self.total_file_path = []  # 所有评论文件的路径
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, idx):
        cur_path = self.total_file_path[idx]
        label_temp = cur_path.split('\\')[-2]
        label = 1 if label_temp == 'pos' else 0
        with open(cur_path,'r',encoding='utf-8') as f:
            text = f.read()
            review = word_tokenize(text)
        input_length = len(review)
        if input_length >= max_len:
            input_length = max_len
        return review,label,input_length

    def __len__(self):
        return len(self.total_file_path)


def collate_fn(batch):
    """

    :param batch: [text,label],[text,label]
    :return:
    """
    batch = sorted(batch,key=lambda x:x[2],reverse=True)
    review,label,input_length = zip(*batch)
    review = [ws.transform(i,max_len=max_len) for i in review]
    review = torch.LongTensor(review)
    label = torch.LongTensor(label)
    input_length = torch.tensor(input_length,dtype=torch.int64)
    return review,label,input_length



train_textdataset = textDataset(mode='train')
trainLoader = DataLoader(train_textdataset,train_batch_size,shuffle=True,collate_fn=collate_fn)
test_textdataset = textDataset(mode='test')
testLoader = DataLoader(test_textdataset,test_batch_size,collate_fn=collate_fn,shuffle=True)


def getdataLoader(mode):
    """
    这个函数用于到时候训练的时候获取dataloader的
    :param mode: 训练还是测试
    :return: DataLoader
    """
    if mode == "train":
        return trainLoader
    else:
        return testLoader



# for idx,(content,label) in enumerate(testLoader):
#     print(label)
# content_list = []
# for content,label in train_textdataset:
#     content_list.append(len(content))
# print(np.mean(content_list))
#33.20325092200519




