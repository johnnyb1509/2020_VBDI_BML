
import numpy as np
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
import string

os.chdir('D:/VBDI_BML')

#%%
# Cleaning data
df_new = pd.read_csv('data/news.csv')
df_new_dir = pd.read_csv('data/news_dir.csv')

data_process = pd.merge(df_new[['NewsId','Ticker', 
                       'NewsTitle', 'NewsFullContent', 'PublicDate']],
                 df_new_dir[['NewsId', 'OrganCode']],
                 on='NewsId', how='left')

data_process['Ticker'].fillna(data_process['OrganCode'], inplace=True)

data_process = data_process.dropna(subset=['NewsFullContent']).reset_index()

def validate_RomanNumerals(string): 
    new_string = [] 
    # Importing regular expression 
    for st in string.split():
        if bool(re.search(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",st)) == False:
            new_string.append(st)
    temp_str = ' '.join(new_string)
    return temp_str

def clean_html(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\n|\t|\r|\xa0|&nbsp;')
    translator = str.maketrans('', '', string.punctuation) # Delete punctuation translator
    for i in range(len(raw_html)):
        print('processing {}%'.format(round(i/len(raw_html)*100, 2)))
        raw_html[i] = re.sub(cleanr,' ', str(raw_html[i])) # clean html syntax
        raw_html[i] = raw_html[i].replace('"', '') # delete quotation marks
        raw_html[i] = re.sub(r'\d+', '', raw_html[i]) # remove numbers
        raw_html[i] = validate_RomanNumerals(raw_html[i]) # remove Roman number
        raw_html[i] = BeautifulSoup(raw_html[i], "lxml").text 
        raw_html[i] = raw_html[i].lower() # chuyen het uppercase sang lowercase
        raw_html[i] = raw_html[i].translate(translator) # delete puntuation
    return raw_html

data_array_clean = clean_html(data_process['NewsFullContent'].values)
data_process['NewsFullContent'] = data_array_clean

print('Saving the processed data to .csv file')
data_process.to_csv('./data/data_preprocess.csv', encoding='utf-8')


#%%
# Content filter on tickers
tickers = ['bid', 'ctg', 'vcb', 'stb', 'ssi', 'vic', 'fpt', 'mwg', 'pnj', 'msn']

list_news = {}
for ticker in tickers:
    print('getting ticker {}'.format(ticker.upper()))
    list_news[ticker] = data_process[data_process['Ticker'] == ticker.upper()].dropna(subset=['PublicDate'])
    print('datefrom: {}'.format(list_news[ticker]['PublicDate'].values[0]))
    print('dateto: {}'.format(list_news[ticker]['PublicDate'].values[-1]))
    print('size: {}'.format(len(list_news[ticker])))
    print('save to {}_news.csv file'.format(ticker))
    list_news[ticker].to_csv('./data/{}_news.csv'.format(ticker))
    print('=============================================')

