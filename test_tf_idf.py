import os
from config import test_pos_root, test_neg_root
import nltk
import pandas as pd
import numpy as np
from config import table_store_path, top_num
from tqdm import trange
import math
from sklearn.metrics import classification_report, roc_auc_score,roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig('ROC.png')

def evaluation(y_true, y_pred, P_pos):
    confusion = confusion_matrix(y_true, y_pred)
    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, ax=ax)
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig('confusion_matrix.png')
    print(classification_report(y_true, y_pred))
    print(roc_auc_score(y_true, P_pos))
    plot_roc(y_true, P_pos)

def test_process(top_num):
    neg_list = os.listdir(test_neg_root)
    neg_list_paths = [os.path.join(test_neg_root, item) for item in neg_list]
    pos_list = os.listdir(test_pos_root)
    pos_list_paths = [os.path.join(test_pos_root, item) for item in pos_list]
    test_doc_list = neg_list_paths + pos_list_paths

    tf_idf_table = pd.read_csv(table_store_path, index_col=[0])
    all_terms = tf_idf_table.index.tolist()
    doc_list = tf_idf_table.columns.tolist()
    doc_list.remove('idf')
    tf_idf_table = np.array(tf_idf_table)

    y_true = []
    y_pred = []
    P_pos = []
    for i in trange(len(test_doc_list)):
        txt_file = test_doc_list[i]
        f = open(txt_file, 'r')
        content = f.read()
        f.close()
        review = nltk.word_tokenize(content)
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

        if pos_votes > neg_votes:
            y_pred.append(1)
        else:
            y_pred.append(0)

        if txt_file.split('\\')[-2] == 'pos':
            y_true.append(1)
        else:
            y_true.append(0)

        P_pos.append(pos_votes/31)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    P_pos = np.array(P_pos)
    evaluation(y_true, y_pred, P_pos)

if __name__ == "__main__":
    # x = list(range(1, 102, 2))
    # y = []
    # for k in range(1, 102, 2):
    #     y.append(test_process(k))

    # plt.plot(x, y)
    # plt.title('The relationship between accuracy top num')
    # plt.xlabel('Top_num')
    # plt.ylabel('Accuracy')
    # plt.savefig('Accuracy-Top_num_relationship.png')

    test_process(top_num)


