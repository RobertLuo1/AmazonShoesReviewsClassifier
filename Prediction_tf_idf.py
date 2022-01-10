#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import nltk
from data_preparation import sub_characters
import pandas as pd
import numpy as np
from config import top_num, table_store_path
from test_tf_idf import tf_generator

def tf_idf_interaction():
    tf_idf_table = pd.read_csv(table_store_path, index_col=[0])
    all_terms = tf_idf_table.index.tolist()
    doc_list = tf_idf_table.columns.tolist()
    doc_list.remove('idf')
    tf_idf_table = np.array(tf_idf_table)

    print("Welcome to the system!")
    input_str = input("Please input your comment:")
    regular_str = sub_characters(input_str)
    while (regular_str == ' '):
        input_str = input("The information for the comment is not that sufficient. Please input your comment again:")
        regular_str = sub_characters(input_str)

    review = nltk.word_tokenize(regular_str)
    vec = tf_generator(review, all_terms)

    results = np.zeros(len(doc_list))
    for j in range(len(doc_list)):
        temp_1 = tf_idf_table[:, 0] * tf_idf_table[:, j + 1]
        temp_2 = tf_idf_table[:, 0] * vec
        if np.linalg.norm(temp_1) > 0:
            temp_1 = temp_1 / np.linalg.norm(temp_1)
        if np.linalg.norm(temp_2) > 0:
            temp_2 = temp_2 / np.linalg.norm(temp_2)
        results[j] = np.dot(temp_1, temp_2)

    result_indexes = results.argsort()[::-1][0:top_num]
    pos_votes = 0
    neg_votes = 0
    for i in result_indexes:
        if doc_list[i].split('\\')[-2] == 'pos':
            pos_votes = pos_votes + 1
        else:
            neg_votes = neg_votes + 1

    if pos_votes > neg_votes:
        print('This is a positive comment!')
    else:
        print('This is a negative comment!')

if __name__ == "__main__":
    tf_idf_interaction()
