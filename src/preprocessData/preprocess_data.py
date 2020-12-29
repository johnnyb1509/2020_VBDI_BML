
import numpy as np
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
import string

os.chdir('D:/VBDI_BML')

#%%
# Load data
df_new = pd.read_csv('data/news.csv')
df_new_dir = pd.read_csv('data/news_dir.csv')

data_process = pd.merge(df_new[['NewsId','Ticker', 
                       'NewsTitle', 'NewsFullContent', 'PublicDate']],
                 df_new_dir[['NewsId', 'OrganCode']],
                 on='NewsId', how='left')

data_process['Ticker'].fillna(data_process['OrganCode'], inplace=True)

data_process = data_process.dropna(subset=['NewsFullContent']).reset_index()

#%%
def validate_RomanNumerals(strng): # xóa số la mã
    new_string = [] 
    # Importing regular expression 
    for st in strng.split():
        if bool(re.search(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",st)) == False:
            new_string.append(st)
    temp_str = ' '.join(new_string)
    return temp_str

def cleaning_dataRaw(dataraw): # cleaning data
    data_array_clean = re.sub(re.compile(r'<[^>]+>'),'',dataraw)
    data_array_clean = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data_array_clean)
    data_array_clean = re.sub(re.compile(r'\n'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r'\t'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r'\r'),'.',data_array_clean)
    data_array_clean = re.sub(re.compile(r'&nbsp;'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r'\xa0'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r'“|”'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r'!'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r'…'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r',|-'),' ',data_array_clean)
    data_array_clean = re.sub(re.compile(r' +'),' ',data_array_clean)
    data_array_clean = data_array_clean.replace('(', ' ').replace(')',' ').replace('.', ' ')
    data_array_clean = re.sub(re.compile(r':'),' ',data_array_clean)
    data_array_clean = data_array_clean.replace('/', ' ').replace('%', ' ').replace(';', ' ')
    return data_array_clean

def del_announcement(data_filtered): # data after filter cac cong bo thong tin giao dich,...
    rows = []
    for index, row in data_filtered.iterrows():
        if row['Ticker']+':' not in row['NewsTitle']:
            rows.append(row.to_frame().transpose())
    data_del = pd.concat(rows)
    return data_del

#%%
# Content filter on tickers
tickers = ['bid', 'ctg', 'vcb', 'stb', 'ssi', 'vic', 'fpt', 'mwg', 'pnj', 'msn']
tickers_out = ['GAS']
tickers = [i.upper() for i in tickers]

#%%
# cleaning data
data_ticker = data_process[data_process['Ticker'].isin(tickers)].dropna(subset=['PublicDate'])
data_ = del_announcement(data_ticker)
data_['NewsFullContent'] = data_['NewsFullContent'].apply(lambda x: cleaning_dataRaw(x))


data_out = data_process[data_process['Ticker'].isin(tickers_out)].dropna(subset=['PublicDate'])
data_out = del_announcement(data_out)
data_out['NewsFullContent'] = data_out['NewsFullContent'].apply(lambda x: cleaning_dataRaw(x))

#%%
# Filter  các trường hợp lỗi data crawl về bị dính liền nhau do các bảng biểu
def sticky_word_table(df): # Filter  các trường hợp lỗi data crawl về bị dính liền nhau do các bảng biểu
    # Tính số từ trong mỗi bài báo, chỉ lấy các bài báo có > 50 từ
    df['len'] = df['NewsFullContent'].apply(lambda text: len(text))
    df = df[df['len'] > 100]
    
    # Tính độ dài của mỗi từ trong bài báo, kiểm tra các bài báo có từ có độ dài > 20
    df['word_len'] = df['NewsFullContent'].apply(lambda text: max([len(w) for w in text.split()]))
    df = df[df['word_len'] < 8]
    df = df[['NewsId', 'Ticker', 'NewsTitle', 'NewsFullContent','PublicDate', 'len', 'word_len']]
    
    df = df.reset_index()
    return df

data_in_cleaned = sticky_word_table(data_)
data_out_cleaned = sticky_word_table(data_out).tail(100)
#%%
data_in_cleaned.to_csv('./data/data_clean_v3.csv')
data_out_cleaned.to_csv('./data/data_clean_outscope.csv')

#%%
#load demo article
demo = pd.read_csv('./data/demo_0.csv').loc[:1]
demo['NewsFullContent'] = demo['NewsFullContent'].apply(lambda x: cleaning_dataRaw(x))
demo.to_csv('./data/demo_clean.csv')
