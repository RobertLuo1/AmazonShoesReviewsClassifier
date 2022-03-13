#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

from config import ws
import numpy as np
import os
import nltk
import math
import pandas as pd
from tqdm import trange
from config import train_neg_root, train_pos_root, table_store_path

def count(term, filename):
    f = open(filename, 'r')
    content = f.read()
    f.close()
    word_list = nltk.word_tokenize(content)
    total = 0
    for word in word_list:
        if term == word:
            total = total+1
    return total

dictionary = ws.count
all_terms = list(dictionary.keys())
term_num = len(all_terms)

neg_list = os.listdir(train_neg_root)
neg_list_paths = [os.path.join(train_neg_root, item) for item in neg_list]
pos_list = os.listdir(train_pos_root)
pos_list_paths = [os.path.join(train_pos_root, item) for item in pos_list]
doc_list = neg_list_paths + pos_list_paths
doc_num = len(doc_list)
doc_list.insert(0, 'idf')

tf_idf_table = np.zeros((term_num, doc_num+1))
for i in trange(term_num):
    for j in range(1, doc_num+1):
        tf_idf_table[i, j] = count(all_terms[i], doc_list[j])
        if tf_idf_table[i, j] > 0:
            tf_idf_table[i, j] = 1 + math.log(tf_idf_table[i, j], 10)

for i in trange(term_num):
    for j in range(1, doc_num+1):
        if tf_idf_table[i, j] != 0:
            tf_idf_table[i, 0] = tf_idf_table[i, 0] + 1
    tf_idf_table[i, 0] = math.log(doc_num/tf_idf_table[i, 0], 10)

data_for_save = pd.DataFrame(tf_idf_table, index=all_terms)
data_for_save.to_csv(table_store_path, header=doc_list)
