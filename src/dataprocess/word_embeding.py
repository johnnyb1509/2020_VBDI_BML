# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:43:47 2020

@author: NguyenSon
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from vncorenlp import VnCoreNLP


annotator = VnCoreNLP("./VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx500m')

def word_segment(text):
    annotated_text = annotator.annotate(text)['sentences'][0]
    return [dic['form'].replace('_',' ').lower() for dic in annotated_text]


# %%
os.chdir('D:/VBDI_BML')
data = pd.read_csv('data/data_labeled_updated.csv')

# %%
X=data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train['NewsFullContent_lower'] = X_train['NewsFullContent'].apply(lambda x: x.lower())
X_train_process = X_train['NewsFullContent_lower'].apply(lambda x: annotator.annotate(str(x)))

#%%
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


# %%
X_train['form'] = X_train_process.apply(lambda x: get_form(x))
X_train['pos_tag'] = X_train_process.apply(lambda x: get_postag(x))
X_train['form_text'] = X_train['form'].apply(lambda x: (" ").join(x))

#%%
# lấy cả bộ data để generate vector, sử dụng fasttext vì lợi thế lớn hơn word2vec
data['NewsFullContent_lower'] = data['NewsFullContent'].apply(lambda x: x.lower())
data_cache = data['NewsFullContent_lower'].apply(lambda x: annotator.annotate(str(x)))
data['form'] = data_cache.apply(lambda x: get_form(x))

#%%
vocab = list(X_train['form'].values)
#%%

from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
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
list_vector = []
for i,r in X_train.iterrows():
    dict_vec = []
    for word in r['form']:
        dict_vec.append(wv[word])
    list_vector.append(dict_vec)

X_train['vector'] = list_vector

X_train['vector'].to_csv('./data/data_train.csv')

#%%
X_test['NewsFullContent_lower'] = X_test['NewsFullContent'].apply(lambda x: x.lower())
X_test_process = X_test['NewsFullContent_lower'].apply(lambda x: annotator.annotate(str(x)))
X_test['form'] = X_test_process.apply(lambda x: get_form(x))

list_vector = []
for i,r in X_test.iterrows():
    dict_vec = []
    for word in r['form']:
        dict_vec.append(wv[word])
    list_vector.append(dict_vec)

X_test['vector'] = list_vector

X_test['vector'].to_csv('./data/data_test.csv')