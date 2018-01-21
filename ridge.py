import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_ridge_regression()


def run_model_ridge_regression():
    X_train = pd.read_pickle('pickle/train_val_all_std.pkl')
    X_val = pd.read_pickle('pickle/val_all_std.pkl')
    y_train = pd.read_pickle('pickle/y_train_val_all_std.pkl')
    y_val = pd.read_pickle('pickle/y_val_all_std.pkl')

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    drop = ['created_at', 'id_str', 'in_reply_to_user_id_str', 'tweetokenize',
            'text', 'pos', 'ner']

    X_train = X_train.drop(drop, axis=1)

    ridge_all_features = ridge(X_train[feat], y_train)
    print('all features accuracy: ', ridge_all_features[0])
    print('all features precision: ', ridge_all_features[1])
    print('all features recall: ', ridge_all_features[2])
    print()

    whole_train = X_train
    ridge_whole = ridge(whole_train, y_train)
    print('whole model accuracy: ', ridge_whole[0])
    print('whole model precision: ', ridge_whole[1])
    print('whole model recall: ', ridge_whole[2])
    print()

    # top_feat = np.load('pickle/top_features.npz')['arr_0'][:100]
    # condensed_train = whole_train[top_feat]
    # ridge_condensed = ridge(condensed_train, y_train)
    # print('condensed model accuracy: ', ridge_condensed[0])
    # print('condensed model precision: ', ridge_condensed[1])
    # print('condensed model recall: ', ridge_condensed[2])
    # print()


def ridge(X_train, y_train):
    # Ridge Logistic Regression

    X = np.array(X_train)
    y = np.array(y_train).ravel()

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model = RidgeClassifier(alpha=10)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test).round()
        y_true = y_test
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict,
                          average='weighted'))
        recalls.append(recall_score(y_true, y_predict, average='weighted'))

    return (np.average(accuracies), np.average(precisions),
            np.average(recalls), model.coef_)


if __name__ == '__main__':
    main()
