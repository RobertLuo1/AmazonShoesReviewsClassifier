#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import torch
import config
from nltk import word_tokenize
from config import ws, max_len, model_file, table_store_path, top_num, test_neg_root, test_pos_root
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from data_preparation import main

def Sigmoid(x):
    return 1/(1+math.exp(-10*(x-0.5)))

def evaluation_final(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, ax=ax)
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig('.\\picture\\confusion_matrix_final.png')
    print(classification_report(y_true, y_pred))

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

def test_final(length):
    standard = StandardScaler()
    neg_list = os.listdir(test_neg_root)
    neg_list_paths = [os.path.join(test_neg_root, item) for item in neg_list]
    pos_list = os.listdir(test_pos_root)
    pos_list_paths = [os.path.join(test_pos_root, item) for item in pos_list]
    test_doc_list = neg_list_paths + pos_list_paths
    device = config.device

    tf_idf_table = pd.read_csv(table_store_path, index_col=[0])
    all_terms = tf_idf_table.index.tolist()
    doc_list = tf_idf_table.columns.tolist()
    doc_list.remove('idf')
    tf_idf_table = np.array(tf_idf_table)

    y_true = []
    y_pred = []
    for i in trange(len(test_doc_list)):
        txt_file = test_doc_list[i]
        f = open(txt_file, 'r')
        content = f.read()
        f.close()
        review = word_tokenize(content)
        input_length = len(review) if len(review) < max_len else max_len
        if input_length > length:
            with torch.no_grad():
                review_encode = ws.transform(review, max_len=max_len)
                review_encode = torch.LongTensor(review_encode)
                review_encode = torch.unsqueeze(review_encode, 0)
                review_encode = review_encode.to(device)
                input_length = torch.unsqueeze(torch.tensor(input_length, dtype=torch.int64), 0)
                model = torch.load(model_file)
                model.eval()
                out = model(review_encode, input_length)
                pred_pro_1 = F.softmax(out, dim=1)[0]
                # pred_pro_1 = out[0]
                pred_pro_1 = pred_pro_1.cpu().numpy().tolist()

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
        if input_length < length:
            choice = np.argmax(pred_pro_2)
            if choice == 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        else:
            pred_pro = np.array([pred_pro_2 + pred_pro_1]).reshape(-1,1)
            pred_pro = standard.fit_transform(pred_pro)
            pred_pro = np.transpose(pred_pro,(1,0))
            with open('./model/svc.pkl', 'rb') as f:
                blending_model = pickle.load(f)
                result = blending_model.predict(pred_pro)
            if result[0] == 1:
                y_pred.append(1)
            else:
                y_pred.append(0)

        if txt_file.split('\\')[-2] == 'pos':
            y_true.append(1)
        else:
            y_true.append(0)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    # np.save('./blendingData/y_predict_svc.npy',y_pred)
    # np.save('./blendingData/y_true_svc.npy',y_true)
    # print('save')
    f1,accuracy,precision,recall = measurement(y_pred,y_true)
    print(f'length is {length}',f'f1 score is {f1:.2f}',f'precision is {precision:.2f}',f'accuracy is {accuracy:.2f}',f'recall is {recall:.2f}')
    return f1,accuracy,precision,recall
    # evaluation_final(y_true, y_pred)

def measurement(y_pred,y_true):
    # y_pred = np.load('./blendingData/y_predict_svc.npy')
    # y_true = np.load('./blendingData/y_true_svc.npy')
    f1 = f1_score(y_true,y_pred)
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    return f1,accuracy,precision,recall

if __name__ == '__main__':
    f1_scores,accuracy_list,precision_list,recall_list = [],[],[],[]
    for i in range(15,50,5):
        f1,accuracy,precision,recall = test_final(i)
        f1_scores.append(f1)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
    plt.plot(f1_scores,label='f1')
    plt.plot(accuracy_list,label='accuracy')
    plt.plot(precision_list,label='precision')
    plt.plot(recall_list,label='recall')
    # plt.xlim(15,50)
    # plt.xticks(np.arange(15,50,5))
    plt.legend()
    plt.savefig('.\\picture\\length_15-50.png')
    # measurement()
