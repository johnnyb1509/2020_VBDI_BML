# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:43:47 2020

@author: NguyenSon
"""
import os
import ast
import pandas as pd
from vncorenlp import VnCoreNLP
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors

os.chdir('D:/VBDI_BML')
annotator = VnCoreNLP("./VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx500m')

def word_segment(text):
    annotated_text = annotator.annotate(text)['sentences'][0]
    return [dic['form'].replace('_',' ').lower() for dic in annotated_text]

def get_postag(x):
    pos = []
    for dic in x['sentences'][0]:
        pos.append(dic['posTag'])
    return pos

def get_form(x):
    pos = []
    for dic in x['sentences'][0]:
        pos.append(dic['form'])
    print(pos)
    return pos

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

#%%
training = pd.read_csv('./data/training.csv')
training['form'] = training['form'].apply(lambda x: ast.literal_eval(x))

testing = pd.read_csv('./data/testing.csv')
testing['form'] = testing['form'].apply(lambda x: ast.literal_eval(x))

#%%
# lấy vocab từ bộ data train
vocab = list(training['form'].values)

#%%
# transform tokenized word to vector
model = FastText()
model.build_vocab(sentences = vocab)
model.train(sentences=vocab, total_examples=len(vocab), epochs=5)
wv = model.wv

# # Saving model
# model.save('fasttext.kvmodel')
# # Load back the same model.
# model2 = KeyedVectors.load('fasttext.kvmodel')

# join new feature to dataframe
def get_vector(data, wv):
    list_vector = []
    for i,r in training.iterrows():
        dict_vec = []
        for word in r['form']:
            dict_vec.append(wv[word])
        list_vector.append(dict_vec)
    return list_vector

list_vector_train = get_vector(training, wv)
training['vector'] = list_vector_train
# training['vector'].to_csv('./data/training_1.csv')

list_vector_test = get_vector(testing, wv)
testing['vector'] = list_vector_test
# testing['vector'].to_csv('./data/testing_1.csv')
