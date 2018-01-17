import pandas as pd
import numpy as np
from src.save_pickle import tf_idf_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def cross_val_data(X_train, X_val, X_test):
    '''
    Takes training, validation,and test sets, concatenates the training data
    and validation sets, and then performs TF-IDF on the ner, text, and pos
    columns. Outputs X_train and X_test sets for features and tf-idf.
    INPUT: DataFrame, DataFrame, DataFrame
    OUTPUT: six DataFrames
    '''

    X_train = pd.concat([X_train, X_val], axis=0)

    # Create TF-IDF for NER column
    tfidf_ner = TfidfVectorizer(ngram_range=(1, 2),
                                lowercase=False,
                                norm='l2',
                                max_df=.99,
                                min_df=.01).fit(X_train['ner'])
    cols = tfidf_ner.get_feature_names()

    X_train_ner = tf_idf_matrix(X_train, 'ner', tfidf_ner, cols)
    X_test_ner = tf_idf_matrix(X_test, 'ner', tfidf_ner, cols)

    # Create TF-IDF for text column
    tfidf_text = TfidfVectorizer(ngram_range=(1, 2),
                                 lowercase=False,
                                 token_pattern='\w+|\@\w+',
                                 norm='l2',
                                 max_df=.99,
                                 min_df=.01).fit(X_train['text'])
    cols = tfidf_text.get_feature_names()

    X_train_tfidf = tf_idf_matrix(X_train, 'text', tfidf_text, cols)
    X_test_tfidf = tf_idf_matrix(X_test, 'text', tfidf_text, cols)

    # Drop ner columns also present in tfidf_text
    columns_to_keep = [x for x in X_train_tfidf if x not in X_train_ner]
    X_train_ner = X_train_ner[columns_to_keep]
    X_test_ner = X_test_ner[columns_to_keep]

    # Create TF-IDF for pos column
    tfidf_pos = TfidfVectorizer(ngram_range=(2, 3),
                                lowercase=False,
                                norm='l2',
                                max_df=.99,
                                min_df=.01).fit(X_train['pos'])
    cols = tfidf_pos.get_feature_names()

    X_train_pos = tf_idf_matrix(X_train, 'pos', tfidf_pos, cols)
    X_test_pos = tf_idf_matrix(X_test, 'pos', tfidf_pos, cols)

    return (X_train, X_train_tfidf, X_train_pos, X_train_ner,
            X_test, X_test_tfidf, X_test_pos, X_test_ner)
