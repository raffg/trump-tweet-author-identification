import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from src.standardize import standardize
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


def main():
    # Load the data
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle('pickle/data_large.pkl')

    # Performing cross-validation, don't need to separate train and validation
    (X_train, X_train_tfidf, X_train_pos, X_train_ner,
     X_test, X_test_tfidf, X_test_pos, X_test_ner) = cross_val_data(X_train,
                                                                    X_val,
                                                                    X_test)
    # Concatenate all training DataFrames
    X_train = pd.concat([X_train, X_train_tfidf,
                         X_train_pos, X_train_ner], axis=1)
    X_test = pd.concat([X_test, X_test_tfidf,
                        X_test_pos, X_test_ner], axis=1)
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

    feat = np.load('all_train_features.npz')['arr_0']

    results = []
    for n in range(1, 25):
        result = svm_grid_search(np.array(X_train[feat[0:n]]),
                                 np.array(y_train).ravel())
        results.append((n, result.best_params_))
        print(n, result.best_params_, result.best_score_)

    for item in results:
        print(item[0], item[1], item[2])


def svm_grid_search(X, y):
    parameters = {'loss': ['hinge', 'log', 'modified_huber',
                  'squared_hinge', 'perceptron'],
                  'penalty': ['l2', 'l1', 'elasticnet'],
                  'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                  'l1_ratio': [0, 0.15, 0.35, 0.5, 0.75, 1],
                  'max_iter': [750, 1000, 1250],
                  'tol': [1e-4, 1e-3, 1e-2]}

    svm = SGDClassifier()
    clf = GridSearchCV(svm, parameters, verbose=True)
    clf.fit(X, y)

    return clf


if __name__ == '__main__':
    main()
