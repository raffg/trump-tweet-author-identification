import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def main():
    # Load the data
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle('pickle/data_large.pkl')

    # Performing cross-validation, don't need to separate train and validation
    X_train = pd.concat([X_train, X_val], axis=0)
    X_train_tfidf = pd.concat([X_train_tfidf, X_val_tfidf], axis=0)
    X_train_pos = pd.concat([X_train_pos, X_val_pos], axis=0)
    X_train_ner = pd.concat([X_train_ner, X_val_ner], axis=0)
    y_train = pd.concat([y_train, y_val], axis=0)

    # Standardize the X data
    feature = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
               'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
               'anticipation', 'disgust', 'fear', 'joy', 'negative',
               'positive', 'sadness', 'surprise', 'trust', 'tweet_length',
               'avg_sentence_length', 'avg_word_length', 'commas',
               'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
               'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
               'all_caps', 'tweetstorm', 'hour', 'period_1', 'period_2',
               'period_3', 'period_4']
    (X_train, X_test) = standardize(feature, X_train, X_test)

    # Add TF-IDF columns to X data
    X_train = pd.concat([X_train, X_train_tfidf,
                         X_train_pos, X_train_ner], axis=1)

    feat = np.load('all_train_features.npz')['arr_0']

    results = []
    for n in range(1, len(feat)):
        result = lr_grid_search(np.array(X_train[feat[0:n]]),
                                np.array(y_train).ravel())
        results.append((n, result.best_params_))
        print(n, result.best_params_, result.best_score_)

    for item in results:
        print(item[0], item[1], item[2])


def lr_grid_search(X, y):
    parameters = {'penalty': ['l1', 'l2'],
                  'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]}
    parameters2 = {'penalty': ['l2'], 'C': [.05, .075, .1, .125, .25]}

    lr = LogisticRegression()
    clf = GridSearchCV(lr, parameters2, verbose=True)
    clf.fit(X, y)

    return clf


if __name__ == '__main__':
    main()
