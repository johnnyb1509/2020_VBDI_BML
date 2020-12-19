#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup


df_new = pd.read_csv('data/news.csv')
df_new_dir = pd.read_csv('data/news_dir.csv')

df_new.shape

data_process = pd.merge(df_new[['NewsId','Ticker', 
                       'NewsTitle', 'NewsFullContent', 'PublicDate']],
                 df_new_dir[['NewsId', 'OrganCode']],
                 on='NewsId', how='left')
data_process.head(20)

data_process.columns
data_process.shape
data_process['Ticker'].fillna(data_process['OrganCode'], inplace=True)
data_process.info()

data_process.head(10)

data_process = data_process.dropna(subset=['NewsFullContent']).reset_index()


data_process['NewsFullContent'][1000]


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\n|\t|\r|\xa0|&nbsp;')
    for i in range(len(raw_html)):
        raw_html[i] = re.sub(cleanr,'', str(raw_html[i]))
        raw_html[i] = BeautifulSoup(raw_html[i], "lxml").text
    return raw_html

data_array_clean = cleanhtml(data_process['NewsFullContent'].values)
data_process['NewsFullContent'] = data_array_clean

data_process.shape

data_process.to_csv('./data/data_preprocess.csv', encoding='utf-8')



