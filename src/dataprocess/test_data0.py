# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 08:55:31 2020

@author: sonnm12
"""

import numpy as np
import pandas as pd

data_news = pd.read_csv("./data/test_news.csv")
data_corp = pd.read_csv("./data/news_dir.csv")

data_news = data_news[['NewsId', 'Ticker', 
                       'NewsTitle', 'NewsFullContent', 'PublicDate']]

data_news = data_news.dropna(subset=['Ticker', 'PublicDate', 'NewsFullContent'])

ticker_count = data_news['Ticker'].value_counts() # count number of mentioned for each ticker  in dataset
#%%
import re
def cleanhtml(raw_html):
    """
    HTML to Clean string
    Parameters
    ----------
    raw_html : TYPE array
        DESCRIPTION.

    Returns
    -------
    raw_html cleaned : TYPE array
        DESCRIPTION.

    """
    for i in range(len(raw_html)):
        raw_html[i] = re.compile('<.*?>').sub('', raw_html[i])
        raw_html[i] = re.compile(r'<[^>]+>').sub('', raw_html[i])
        raw_html[i] = raw_html[i].replace('&nbsp;','')
        raw_html[i] = re.compile('\r\n').sub(' ', raw_html[i])
    return raw_html

#%%
data_array_clean = cleanhtml(data_news['NewsFullContent'].values)
data_news['FullContent_clean'] = data_array_clean


#%%
# Bags of word
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_array_clean)

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

