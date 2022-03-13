#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

from  wordtosequence import wordToSequence
import os
from tqdm import tqdm
from nltk import word_tokenize
import pickle

def saveDisctionary():
    # dict保存
    data_base_file = r'../data/dataset'
    for category in ['train', 'test']:
        review_path = os.path.join(data_base_file, category)
        temp_review_path = [os.path.join(review_path, 'pos'), os.path.join(review_path, 'neg')]
        for review_path in temp_review_path:
            file_names = os.listdir(review_path)
            file_paths = [os.path.join(review_path, file_name) for file_name in file_names if file_name.endswith('txt')]
            for file_path in tqdm(file_paths, total=len(file_paths), ascii=True, desc='Building vocabulary'):
                with open(file_path, 'r', encoding='UTF-8') as f:
                    text = f.read()
                    review = word_tokenize(text)
                ws.fit(review)

    ws.build_vocabulary(min=10, max=3000, max_feature=10000)
    model_file = os.path.join('./model', 'ws1.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(ws, f)
    print(len(ws))
    print('Finish')





if __name__ == '__main__':
    ws = wordToSequence()
    # saveDisctionary()
    # dictionary_path = os.path.join(r'./model', 'ws1.pkl')
    # with open(dictionary_path, 'rb') as f:
    #     ws = pickle.load(f)









