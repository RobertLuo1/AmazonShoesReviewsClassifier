#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import os
import pickle
# from tqdm import tqdm  # 用于打印进度条
# from nltk import word_tokenize

class wordToSequence():
    """
    将每一个词语映射成一个对应的数字来放入神经网络之中
    """
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 1
    PAD = 0#这里写0，方便之后的操作，记住跑一遍

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}

    def fit(self,sentence):
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1

    def build_vocabulary(self,min,max,max_feature):
        if min is not None:
            #将词频少的丢弃
            self.count = {key: value for key, value in self.count.items() if value > min}
        if max is not None:
            #将词频太高的也丢弃（这里丢弃的可能是is，are这些对结果没有意义的词语）
            self.count = {key: value for key, value in self.count.items() if value < max}
            # 限制保留词语数
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_feature])  # 将字典排序

        for word in self.count:
            #将词语加入字典中
            self.dict[word] = len(self.dict)

    def transform(self,sentence,max_len):
        length = len(sentence)
        if length < max_len:
            sentence = sentence+[self.PAD_TAG]*(max_len-length)
        else:
            sentence = sentence[:max_len]
        sequence = [self.dict.get(word,self.UNK) for word in sentence]
        return sequence

    def __len__(self):
        return len(self.dict)






