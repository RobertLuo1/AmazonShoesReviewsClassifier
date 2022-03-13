import numpy as np
import pandas as pd
import re
import os
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import config

def list_sub(list_1, list_2):
    temp_list = []
    for e in list_1:
        if e not in list_2:
            temp_list.append(e)
    return temp_list

def delete_space(list):
    flag = 0
    for i in range(len(list)):
        if list[i] != ' ':
            flag = 1

    if flag == 0:
        return []

    if list[0] == ' ':
        i = 0
        while (list[i] == ' '):
            list[i] = ''
            i = i + 1

    if list[-1] == ' ':
        i = -1
        while (list[i] == ' '):
            list[i] = ''
            i = i - 1

    for i in range(len(list)):
        if list[i] == ' ':
            j = i + 1
            while (list[j] == ' '):
                list[j] = ''
                j = j + 1

    return list

def get_items(list, target):
    temp = []
    temp_ex = []
    for i in range(4):
        temp_ex.append(0)
    temp.append(temp_ex)
    temp = np.array(temp,dtype=object)

    for i in range(list.shape[0]):
        if (list[i, 2] == target):
            temp = np.insert(temp, temp.shape[0], list[i,:], axis = 0)
    temp = np.delete(temp,0,axis = 0)

    return temp

def get_items_2(total_data, indexes, path):
    for i in range(len(indexes)):
        file_name = path + '\\' + str(i) + '.txt'
        recorder = open(file_name, 'a+')
        recorder.write(total_data[indexes[i], 0])
        recorder.close()


def sub_characters(str):
    temp = str.lower()
    temp = re.sub(r'won\'t', 'will not', temp)
    temp = re.sub(r'\'s', ' is', temp)
    temp = re.sub(r'n\'t', ' not', temp)
    temp = re.sub(r'\'m', ' am', temp)
    temp = re.sub(r'\'ll', ' will', temp)
    temp = re.sub(r'\'ve', ' have', temp)
    temp = re.sub(r'\'re', ' are', temp)
    temp = re.sub(r'\'d', ' would', temp)
    temp = re.sub(r'won’t', 'will not', temp)
    temp = re.sub(r'’s', ' is', temp)
    temp = re.sub(r'n’t', ' not', temp)
    temp = re.sub(r'’m', ' am', temp)
    temp = re.sub(r'’ll', ' will', temp)
    temp = re.sub(r'’ve', ' have', temp)
    temp = re.sub(r'’re', ' are', temp)
    temp = re.sub(r'’d', ' would', temp)
    temp = re.sub(r'[^a-zA-Z]', ' ', temp)

    if len(temp) == 1:
        return ' '

    result = []
    for item in temp:
        result.append(item)

    if result[1] == ' ':
        if result[0] != 'a' and result[0] != 'i':
            result[0] = ' '

    if result[-2] == ' ':
        if result[-1] != 'a' and result[-1] != 'i':
            result[-1] = ' '

    i = 1
    while (i < len(result)-1):
        if result[i-1] == ' ' and result[i+1] == ' ':
            if result[i] != 'a' and result[i] != 'i':
                result[i] = ' '
        i = i + 1

    result = delete_space(result)
    result = "".join(result)

    result = result.split()
    # result = [w for w in result if w not in stopwords.words("english")]
    item_list = ['n', 'v', 'a', 'r']
    for i in range(len(item_list)):
        result = [WordNetLemmatizer().lemmatize(w, pos=item_list[i]) for w in result]

    if len(result) == 0:
        return ' '

    final = ''
    for i in range(len(result)-1):
        final = final + result[i] + ' '
    final = final + result[-1]
    return final

def main():

    if os.path.exists('dataset') != True:
        os.makedirs('dataset\\train\\pos')
        os.makedirs('dataset\\train\\neg')
        os.makedirs('dataset\\test\\pos')
        os.makedirs('dataset\\test\\neg')

    total_data = pd.read_csv(config.review_union)
    total_data = np.array(total_data)

    i = 0
    while (i < total_data.shape[0]):
        total_data[i, 0] = sub_characters(total_data[i, 0])
        if total_data[i, 0] == ' ':
            total_data = np.delete(total_data, i, 0)
            continue
        print(i)
        print(total_data[i, 0])
        i = i+1

    ratio = 0.2

    postive_data = get_items(total_data, 1)
    negative_data = get_items(total_data, 0)

    num_pos = postive_data.shape[0]
    num_neg = negative_data.shape[0]

    indexes_pos = list(range(num_pos))
    indexes_neg = list(range(num_neg))

    test_pos = int(num_pos * 0.2)
    test_neg = int(num_neg * 0.2)
    test_pos_indexes = random.sample(range(num_pos), test_pos)
    test_neg_indexes = random.sample(range(num_neg), test_neg)

    train_pos_indexes = list_sub(indexes_pos, test_pos_indexes)
    train_neg_indexes = list_sub(indexes_neg, test_neg_indexes)

    get_items_2(postive_data, test_pos_indexes, 'dataset\\test\\pos')
    get_items_2(negative_data, test_neg_indexes, 'dataset\\test\\neg')
    get_items_2(postive_data, train_pos_indexes, 'dataset\\train\\pos')
    get_items_2(negative_data, train_neg_indexes, 'dataset\\train\\neg')


if __name__ == "__main__":
    main()
