# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:20:10 2020

@author: NguyenSon
"""

import numpy as np
import pandas as pd
import os

os.chdir('D:/shan_github/2020_VBDI_BML')

data_processed = pd.read_csv("./data/data_preprocess.csv")

#%%
tickers = ['bid', 'ctg', 'vcb', 'stb', 'ssi', 'vic', 'fpt', 'mwg', 'pnj', 'msn']

list_news = {}
for ticker in tickers:
    print('getting ticker {}'.format(ticker))
    list_news[ticker] = data_processed[data_processed['Ticker'] == ticker.upper()].dropna(subset=['PublicDate'])
    print('datefrom: {}'.format(list_news[ticker]['PublicDate'].values[0]))
    print('dateto: {}'.format(list_news[ticker]['PublicDate'].values[-1]))
    print('size: {}'.format(len(list_news[ticker])))
    print('save to {}_news.csv file'.format(ticker))
    list_news[ticker].to_csv('./data/{}_news.csv'.format(ticker))
    print('=============================================')
    