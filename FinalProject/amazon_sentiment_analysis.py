import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import langid
import os
import kaleido
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, \
    precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
import plotly
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import nltk
from nltk.corpus import stopwords
#import spacy
import matplotlib.pyplot as plt

"""
Dataset downloaded from: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
and extract test.csv and train.csv into the 'data' folder
"""


def stopwords_processing():
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    print(stop_words)


    # remove tokens ending in 'n' or 'n't'
    for a in [l for l in stop_words if l.endswith('n') or l.endswith("n't")][12:]:
        stop_words.remove(a)

    stop_words.remove('no')
    stop_words.remove('nor')
    stop_words.remove('not')
    stop_words.remove('against')
    stop_words.remove('down')
    stop_words.remove('up')
    return stop_words


if __name__ == "__main__":

    train_df = pd.read_csv('FinalProject/data/train.csv', header=None, names=['label', 'title', 'text'])
    test_df = pd.read_csv('FinalProject/data/test.csv', header=None, names=['label', 'title', 'text'])

    print(train_df.head())

    print("\nsize of training set:", len(train_df))
    print("\nsize of test set:", len(test_df))

    train_len = 50000
    test_len = 10000
    rs = 42

    # merge train_df with test_df, keep 50000 and 10000 samples for each of them respectively
    df = pd.concat([train_df.loc[train_df['label'] == 1].sample(train_len // 2, random_state=rs),
                    train_df.loc[train_df['label'] == 2].sample(train_len // 2, random_state=rs),
                    test_df.loc[test_df['label'] == 1].sample(test_len // 2, random_state=rs),
                    test_df.loc[test_df['label'] == 2].sample(test_len // 2, random_state=rs)]).reset_index(drop=True)

    # fill NA/NaN values of column 'title' with ''
    df['title'].fillna('', inplace=True)
    # merge the title and text columns
    df['text'] = df['title'] + '. ' + df['text']
    df.drop('title', axis=1, inplace=True)

    df.head()

    print(f'Label counts - training set:\n{df[:train_len].label.value_counts()}')
    print(f'\nLabel counts - test set:\n{df[train_len:].label.value_counts()}')

    langs = [langid.classify(s)[0] for s in df['text']]
    Counter(langs)

    # remove non-english reviews from dataset
    df = df[[l == 'en' for l in langs]]

    df.to_csv("FinalProject/data/text_title.csv", index=False)

    if not os.path.exists('images'):
        os.mkdir('images')


    cv = CountVectorizer(ngram_range=(1, 3))
    cv.fit(df['text'])
    print(f"Number of n-grams (n=1,2,3) in the corpus: {len(cv.vocabulary_)}")

    cv_1 = CountVectorizer(ngram_range=(2, 3))
    cv_1.fit(df['text'])
    print(f"Number of n-grams (n=2,3) in the corpus: {len(cv_1.vocabulary_)}")

    fts = cv.get_feature_names()

    freq = cv.transform(df['text'])
    gram_counts = np.array(freq.sum(0)).squeeze()

    gram_counts_df = pd.DataFrame({'n-gram': [fts[i] for i in gram_counts.argsort()], 'count': sorted(gram_counts)})

    x = px.bar(gram_counts_df[-20:], y='n-gram', x='count', title='Most frequent n-grams (n=1,2,3)',
           orientation='h', height=600)
    x.show()
    #x.write_image('images/Most frequent n-grams (n=1,2,3).png')


    # returns array of transformed feature names.
    fts_1 = cv_1.get_feature_names()

    freq_1 = cv_1.transform(df['text'])
    gram_counts_1 = np.array(freq_1.sum(0)).squeeze()

    gram_counts_df_1 = pd.DataFrame({'n-gram': [fts_1[i] for i in gram_counts_1.argsort()], 'count': sorted(gram_counts_1)})

    x = px.bar(gram_counts_df_1[-20:], y='n-gram', x='count', title='Most frequent n-grams (n=2,3)', orientation='h',
           height=600)
    x.show()
    #x.write_image('images\Most frequent n-grams (n=2,3).png')

    stop_words = stopwords_processing()

    print(stop_words)

    cv = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words)
    cv.fit(df['text'])
    print(f"Number of ngrams (n=1,2,3) in the corpus without stopwords: {len(cv.vocabulary_)}")

    cv_1 = CountVectorizer(ngram_range=(2, 3), stop_words=stop_words)
    cv_1.fit(df['text'])
    print(f"Number of ngrams (n=2,3) in the corpus without stopwords: {len(cv_1.vocabulary_)}")



    fts = cv.get_feature_names()
    freq = cv.transform(df['text'])
    gram_counts = np.array(freq.sum(0)).squeeze()

    gram_counts_df = pd.DataFrame({'n-gram': [fts[i] for i in gram_counts.argsort()],
                                   'count': sorted(gram_counts)})

    x =px.bar(gram_counts_df[-20:], y='n-gram', x='count', title='Most frequent n-grams (n=1,2,3) without stopwords',
           orientation='h', height=600)
    x.show()
    #x.write_image('images\Most frequent n-grams (n=1,2,3) without stopwords.png')

    fts_1 = cv_1.get_feature_names()
    freq_1 = cv_1.transform(df['text'])
    gram_counts_1 = np.array(freq_1.sum(0)).squeeze()

    gram_counts_df_1 = pd.DataFrame({'n-gram': [fts_1[i] for i in gram_counts_1.argsort()], 'count': sorted(gram_counts_1)})

    x = px.bar(gram_counts_df_1[-20:], y='n-gram', x='count', title='Most frequent n-grams (n=2,3) without stopwords',
           orientation='h', height=600)
    x.show()
    #x.write_image('images/Most frequent n-grams (n=2,3) without stopwords.png')



    freq_pos = cv.transform(df.loc[df['label'] == 2, 'text'])
    gram_counts_pos = np.array(freq_pos.sum(0)).squeeze()

    gram_counts_df_pos = pd.DataFrame(
        {'n-gram': [fts[i] for i in gram_counts_pos.argsort()], 'count': sorted(gram_counts_pos)})

    x = px.bar(gram_counts_df_pos[-20:], y='n-gram', x='count',
           title='Most frequent n-grams (n=1,2,3) without stopwords - Positive reviews', orientation='h', height=600)
    x.show()
    #x.write_image('images/Most frequent n-grams (n=1,2,3) without stopwords - Positive reviews.png')

    freq_neg = cv.transform(df.loc[df['label'] == 1, 'text'])
    gram_counts_neg = np.array(freq_neg.sum(0)).squeeze()

    gram_counts_df_neg = pd.DataFrame(
        {'n-gram': [fts[i] for i in gram_counts_neg.argsort()], 'count': sorted(gram_counts_neg)})

    x = px.bar(gram_counts_df_neg[-20:], y='n-gram', x='count',
           title='Most frequent n-grams (n=1,2,3) without stopwords - Negative reviews', orientation='h', height=600)
    x.show()
    #x.write_image('images/Most frequent n-grams (n=1,2,3) without stopwords - Negative reviews.png')



    freq_1_pos = cv_1.transform(df.loc[df['label'] == 2, 'text'])
    gram_counts_1_pos = np.array(freq_1_pos.sum(0)).squeeze()

    gram_counts_df_1_pos = pd.DataFrame({'n-gram': [fts_1[i] for i in gram_counts_1_pos.argsort()],
                                         'count': sorted(gram_counts_1_pos)})

    x = px.bar(gram_counts_df_1_pos[-20:], y='n-gram', x='count',
           title='Most frequent n-grams (n=2,3) without stopwords - Positive reviews', orientation='h', height=600)
    x.show()
    #x.write_image('images/Most frequent n-grams (n=2,3) without stopwords - Positive reviews.png')


    freq_1_neg = cv_1.transform(df.loc[df['label'] == 1, 'text'])
    gram_counts_1_neg = np.array(freq_1_neg.sum(0)).squeeze()

    gram_counts_df_1_neg = pd.DataFrame(
        {'n-gram': [fts_1[i] for i in gram_counts_1_neg.argsort()], 'count': sorted(gram_counts_1_neg)})

    x = px.bar(gram_counts_df_1_neg[-20:], y='n-gram', x='count',
           title='Most frequent n-grams (n=2,3) without stopwords - Negative reviews', orientation='h', height=600)
    x.show()
    #x.write_image('images/Most frequent n-grams (n=2,3) without stopwords - Negative reviews.png')

