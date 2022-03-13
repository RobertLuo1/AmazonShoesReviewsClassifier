#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import os
import pickle
import torch

#用于爬虫的参数
Chrome_PATH = r'D:\chrome_driver\chromedriver.exe'
amazon_url = 'https://www.amazon.com'
base_url = "https://www.amazon.com/s?k="
# search_query = "nike+shoes+men"
# search_query = 'nike+shoes+women'
search_query = 'adidas+shoes+men'
data_file = './original_data/reviews_man_neg_2.csv'
reviews_number = 500


##数据处理
review_union = r'./tools/reviews_union.csv'

#用于构建数据集的参数
train_batch_size = 64
test_batch_size = 128

#用于构建字典的参数
dictionary_path = os.path.join(r'./model/ws1.pkl')
with open(dictionary_path,'rb') as f:
    ws = pickle.load(f)

min_count = 10
max_count = 3000
max_feature = 10000
max_len = 80


#构建神经网络的参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_size = 256
hidden_layer = 300
num_layer = 2
learning_rate = 0.001
epoch = 60

#外部接口
model_file = './model/Bilstm_models60 29.54.pth'

#运用tf_idf_table前需要的准备
top_num = 31
train_neg_root = '..\\data\\dataset\\train\\neg'
train_pos_root = '..\\data\\dataset\\train\\pos'
table_store_path = '.\\tools\\tf_idf_table.csv'
test_neg_root = '..\\data\\dataset\\test\\neg'
test_pos_root = '..\\data\\dataset\\test\\pos'
