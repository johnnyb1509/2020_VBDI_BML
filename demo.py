import pandas as pd
from vncorenlp import VnCoreNLP
import ast
import pycrfsuite
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

# EXTRACT NER FEATURE
def word_segment(text, annotator):
    annotated_text = annotator.annotate(text)['sentences'][0]
    return [dic['form'].replace('_',' ').lower() for dic in annotated_text]

def vnner(text, ner):
    annotated_text = ner.annotate(text)
    pos = []
    for dic in annotated_text['sentences'][0]:
        if dic['nerLabel'] != 'O':
            pos.append(dic['form'].replace('_',' ').lower())
        
    return list(set(pos))

def tag(word, dic):
    if word in dic:
        return True
    else:
        return False

def ner_tag(row, ner):
    keywords = vnner(row['NewsFullContent'], ner)
    text = row['word_segment']
    text_df = pd.DataFrame(text)
    text_df.columns = ['word']

    text_df['tag'] = text_df['word'].apply(lambda x: tag(x, keywords))
    
    # return list(zip(text_df['word'], text_df['tag']))
    return list(text_df['tag'])

def extract_ner(data, annotator, ner):
    df = data.copy()
    df['word_segment'] = df['NewsFullContent'].apply(lambda text: word_segment(text, annotator))

    name_entity_reg = []
    for i,row in df.iterrows():
        name_entity_reg.append(ner_tag(row, ner))

    df['name_entity'] = name_entity_reg

    return df['name_entity']

# EXTRACT POSITION EMBEDDING FEATURE
def title_detect(row, annotator):
    annotated_text = annotator.annotate(row['NewsTitle'])['sentences'][0]
    title = [dic['form'].replace('_',' ').lower() for dic in annotated_text]
    
    text = row['word_segment']
    text_df = pd.DataFrame(text)
    text_df.columns = ['word']

    text_df['tag'] = text_df['word'].apply(lambda x: tag(x, title))
    
    return list(text_df['tag'])

def extract_title(data, annotator):
    df = data.copy()
    df['word_segment'] = df['NewsFullContent'].apply(lambda text: word_segment(text, annotator))

    title_tag = []
    for i,row in df.iterrows():
        title_tag.append(title_detect(row, annotator))

    df['title_tag'] = title_tag

    return df['title_tag']

# EXTRACT POS TAG AND TFIDF FEATURE
def get_postag(x):
    pos = []
    for dic in x['sentences'][0]:
        pos.append(dic['posTag'])
    return pos

def get_form(x):
    pos = []
    for dic in x['sentences'][0]:
        pos.append(dic['form'])
    return pos

def get_tfidf_for_word(word, dic):
    if word in dic.keys():
        return dic[word]
    else:
        return 0

def get_tfidf_for_docs(text, dic):
    text_df = pd.DataFrame(text)
    text_df.columns = ['word']
    tfidf_words = text_df['word'].apply(lambda x: get_tfidf_for_word(x, dic))
    return tfidf_words

def get_ifidf_for_words(text, tfidf):
    tfidf_matrix= tfidf.transform([text]).todense()
    feature_index = tfidf_matrix[0,:].nonzero()[1]
    feature_names = tfidf.get_feature_names()
    tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
    return dict(tfidf_scores)


# COMBINE FEATURE

def data_process_test(X, annotator, pos, ner): 
    # X = X['NewsFullContent'].apply(lambda x: pos.annotate(str(x)))
    X_t = X.copy()  
    X_t = X_t['NewsFullContent'].apply(lambda x: pos.annotate(str(x)))
    X['form'] = X_t.apply(lambda x: get_form(x))
    X['pos_tag'] = X_t.apply(lambda x: get_postag(x))
    X['form_text'] = X['form'].apply(lambda x: (" ").join(x))
    tfidf_save = pickle.load(open('C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/model/tfidf.sav', 'rb'))
    # tfidf_matrix = tfidf_save.transform(X['form_text'])
    # feature_names = tfidf_save.get_feature_names()
    X['ifidf_for_words'] = X['form_text'].apply(lambda x : get_ifidf_for_words(x, tfidf_save))
    tfidf_docs = []
    for i,row in X[['form','ifidf_for_words']].iterrows():
        tfidf_docs.append(get_tfidf_for_docs(row['form'],row['ifidf_for_words']))
    X['tfidf_docs'] = tfidf_docs
    X['name_entity'] = extract_ner(X, annotator, ner)
    X['title_tag'] = extract_title(X, annotator)

    X = X.reset_index()
    return X


def combine_feature(data):
    df = data.copy()
    features = []
    for i in range(len(df)):
        form = pd.DataFrame(df['form'][i])
        
        pos_tag = pd.DataFrame(df['pos_tag'][i])
        tfidf_docs = pd.DataFrame(df['tfidf_docs'][i])
        name_entity = pd.DataFrame(df['name_entity'][i])
        title_tag = pd.DataFrame(df['title_tag'][i])
        df_col_merged = pd.concat([form, pos_tag,tfidf_docs,name_entity,title_tag], axis=1)
        feature = df_col_merged.values.tolist()
        features.append(feature)
    df['features'] = features

    return df['features']

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    tfidf = doc[i][2]
    ner = doc[i][3]
    title = doc[i][4]
    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isner=%s' % ner,
        'tfidf=%f' % tfidf,
        'word.istitle=%s' % title,
        'postag=' + postag
    ]

    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        tfidf1 = doc[i-1][2]
        ner1 = doc[i-1][3]
        title1 = doc[i-1][4]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.isner=%s' % ner1,
            '-1:tfidf=%f' % tfidf1,
            '-1:word.istitle=%s' % title1,
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        tfidf1 = doc[i+1][2]
        ner1 = doc[i+1][3]
        title1 = doc[i+1][4]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.isner=%s' % ner1,
            '+1:tfidf=%f' % tfidf1,
            '+1:word.istitle=%s' % title1,
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# VISUALIZE
def test_result(X_test, X_test1, y_pred, row_to_check = 0):
    x_ = X_test1[row_to_check]
    y_ = y_pred[row_to_check]
    x_word = []
    for i in range(len(x_)):
        x_word.append(x_[i][1].split(sep='=')[-1])
    df = pd.DataFrame({'X_test': x_word, 'y_pred': y_})
    df = df[df['y_pred'] == 'I']
    key_extract = df['X_test'].values
    full_content = X_test['NewsFullContent'].loc[row_to_check]
    title = X_test['NewsTitle'].loc[row_to_check]
    return title, full_content, key_extract

if __name__ == "__main__":
    df = pd.read_csv('C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/data/demo_clean.csv')

    annotator = VnCoreNLP("C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
    pos = VnCoreNLP("C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx500m')
    ner = VnCoreNLP("C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg,pos,ner", max_heap_size='-Xmx2g')

    df['features'] = combine_feature(data_process_test(df, annotator, pos, ner))

    feat = df['features'].tolist()
    X = [extract_features(doc) for doc in feat]

    model = pickle.load(open('C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/model/crf.sav', 'rb'))

    y_pred = model.predict(X)
    # Create a mapping of labels to indices
    labels = {"N": 0, "I": 1}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])

    title, full_content, key_extract = test_result(df, X, y_pred, row_to_check = 0)
    with open('C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/result/test1.txt', "w", encoding="utf-8") as f:
        f.write("Title: \n")
        f.write(title)
        f.write("\n Full content: \n")
        f.write(full_content)
        f.write("\n Keyword: \n")
        f.write(str(key_extract))

    title, full_content, key_extract = test_result(df, X, y_pred, row_to_check = 1)
    with open('C:/Users/anhdq33/Downloads/VinBigData/ML/Project/2020_VBDI_BML/result/test2.txt', "w", encoding="utf-8") as f:
        f.write("Title: \n")
        f.write(title)
        f.write("\n Full content: \n")
        f.write(full_content)
        f.write("\n Keyword: \n")
        f.write(str(key_extract))


    



