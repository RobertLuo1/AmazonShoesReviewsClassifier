#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import torch
import config
from data_preparation import sub_characters
from nltk import word_tokenize
from config import ws, max_len, model_file, table_store_path, top_num
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import re

def Sigmoid(x):
    """
    The sigmoid-likewise function
    """
    return 1/(1+math.exp(-10*(x-0.5)))

def tf_generator(review, all_terms):
    vec_len = len(all_terms)
    vector = np.zeros(vec_len)
    for word in review:
        if word in all_terms:
            vector[all_terms.index(word)] = vector[all_terms.index(word)] + 1
    for i in range(vec_len):
        if vector[i] > 0:
            vector[i] = math.log(vector[i], 10) + 1
    return vector

def rule(sentence):
    """
    construct some rules
    """
    negative_pattern = []
    negative_pattern.append(re.compile(r'.*not [like|love]((?!(but|But|However|however)).)*$'))
    negative_pattern.append(re.compile(r'.* not buy .*'))
    negative_pattern.append(re.compile(r'.* not (.*) buy .*'))
    for pattern in negative_pattern:
        match = re.search(pattern,sentence)
        if match:
            return True
    return False

def interaction():
    """
    The interaction system
    """
    device = config.device
    print("*"*80)
    print("Welcome to the system!")
    input_str = ''
    while input_str != 'q':
        print("-"*80)
        input_str = input("Please input your comment or pleas q to quit:\n")
        if input_str == 'q':
            print("Thank you for choosing our system.")
            print("We hope the system help and wish you have a nice day!")
            print("*"*80)
            break
        judge = rule(sentence=input_str)
        if judge:
            print('This is a negative comment!')
            continue
        regular_str = sub_characters(input_str)
        while (regular_str == ' '):
            input_str = input("The information for the comment is not that sufficient. Please input your comment again:")
            regular_str = sub_characters(input_str)

        review = word_tokenize(regular_str)
        input_length = len(review) if len(review) < max_len else max_len
        if input_length >= 30:
            with torch.no_grad():            
                review_encode = ws.transform(review, max_len=max_len)
                review_encode = torch.LongTensor(review_encode)
                review_encode = torch.unsqueeze(review_encode, 0)
                review_encode = review_encode.to(device)
                input_length = torch.unsqueeze(torch.tensor(input_length, dtype=torch.int64), 0)
                model = torch.load(model_file)
                model.eval()
                out = model(review_encode, input_length)
                # pred_pro_1 = out[0]
                pred_pro_1 = F.softmax(out, dim=1)[0]
                pred_pro_1 = pred_pro_1.cpu().numpy().tolist()

        tf_idf_table = pd.read_csv(table_store_path, index_col=[0])
        all_terms = tf_idf_table.index.tolist()
        doc_list = tf_idf_table.columns.tolist()
        doc_list.remove('idf')
        tf_idf_table = np.array(tf_idf_table)

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
        for j in result_indexes:
            if doc_list[j].split('\\')[-2] == 'pos':
                pos_votes = pos_votes + 1
            else:
                neg_votes = neg_votes + 1

        pred_pro_2 = [Sigmoid(pos_votes/top_num), Sigmoid(neg_votes/top_num)]
        if input_length < 30:
            choice = np.argmax(pred_pro_2)
            result = choice
            if result == 0:
                print('This is a positive comment!')
            else:
                print('This is a negative comment!')
        else:
            pred_pro = np.array([pred_pro_2 + pred_pro_1]).reshape(-1,1)
            pred_pro = np.transpose(StandardScaler().fit_transform(pred_pro),(1,0))
            with open('./model/svc.pkl', 'rb') as f:
                blending_model = pickle.load(f)
                result = blending_model.predict(pred_pro)[0]
            if result == 1:
                print('This is a positive comment!')
            else:
                print('This is a negative comment!')

if __name__ == '__main__':
    interaction()
