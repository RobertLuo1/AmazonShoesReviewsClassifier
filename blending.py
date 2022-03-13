#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import os
import torch
from config import train_neg_root, train_pos_root,ws,max_len,device,model_file
import nltk
import pandas as pd
import numpy as np
from config import table_store_path, top_num
from tqdm import trange
import math
import torch.nn.functional as F
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def Sigmoid(x):
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

def blend_process(top_num):
    standstandard = StandardScaler()
    label_list = []
    neg_list = os.listdir(train_neg_root)
    neg_list_paths = [os.path.join(train_neg_root, item) for item in neg_list]
    pos_list = os.listdir(train_pos_root)
    pos_list_paths = [os.path.join(train_pos_root, item) for item in pos_list]
    train_doc_list = pos_list_paths+neg_list_paths

    tf_idf_table = pd.read_csv(table_store_path, index_col=[0])
    all_terms = tf_idf_table.index.tolist()
    doc_list = tf_idf_table.columns.tolist()
    doc_list.remove('idf')
    tf_idf_table = np.array(tf_idf_table)
    P_pos = []

    for i in trange(len(train_doc_list)):
        txt_file = train_doc_list[i]
        label = 1 if txt_file.split('\\')[-2] == 'pos' else 0
        label_list.append(label)
        f = open(txt_file, 'r')
        content = f.read()
        f.close()
        review = nltk.word_tokenize(content)

        with torch.no_grad():
            input_length = len(review) if len(review) < max_len else max_len
            review_encode = ws.transform(review, max_len=max_len)
            review_encode = torch.LongTensor(review_encode)
            review_encode = torch.unsqueeze(review_encode, 0)
            review_encode = review_encode.to(device)
            input_length = torch.unsqueeze(torch.tensor(input_length, dtype=torch.int64), 0)
            model = torch.load(model_file)
            model.eval()
            out = model(review_encode, input_length)
            # pred_pro = out[0]
            pred_pro = F.softmax(out, dim=1)[0]
            # pred_pro = out.max(dim=-1)[-1]
            pred_pro = pred_pro.cpu().numpy().tolist()

        vec = tf_generator(review, all_terms)

        results = np.zeros(len(doc_list))
        for j in range(len(doc_list)):
            temp_1 = tf_idf_table[:, 0] * tf_idf_table[:, j + 1]
            temp_2 = tf_idf_table[:, 0] * vec
            if np.linalg.norm(temp_1) > 0:
                temp_1 = temp_1/np.linalg.norm(temp_1)
            if np.linalg.norm(temp_2) > 0:
                temp_2 = temp_2/np.linalg.norm(temp_2)
            results[j] = np.dot(temp_1, temp_2)

        result_indexes = results.argsort()[::-1][0:top_num]
        pos_votes = 0
        neg_votes = 0
        for j in result_indexes:
            if doc_list[j].split('\\')[-2] == 'pos':
                pos_votes = pos_votes + 1
            else:
                neg_votes = neg_votes + 1

        temp = np.array([Sigmoid((pos_votes/top_num)), Sigmoid((neg_votes/top_num)),pred_pro[0],pred_pro[1]]).reshape(-1,1)
        P_pos.append(standstandard.fit_transform(temp))

    P_pos = np.array(P_pos)
    label_list = np.array(label_list)
    return P_pos,label_list

def saveData():
    X,y=blend_process(top_num)
    np.save('./blendingData/blending_X_new.npy',X)
    np.save('./blendingData/blending_y_new.npy',y)
    print('Finish')

def blending_model():
    X = np.load('./blendingData/blending_X_new.npy')
    X = np.squeeze(X,axis=-1)
    y = np.load('./blendingData/blending_y_new.npy')
    model = SVC()
    # model = LogisticRegression()
    # model  =RandomForestClassifier()
    model.fit(X,y)
    with open('./model/svc.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    # saveData()
    blending_model()