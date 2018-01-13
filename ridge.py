import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_ridge_regression('pickle/data.pkl')


def run_model_ridge_regression(file):
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle(file)

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'negative', 'neutral', 'positive', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'period_1', 'period_2',
            'period_3', 'period_4']

    X_train = pd.concat([X_train, X_val], axis=0)
    (X_train, X_test) = standardize(feat, X_train, X_test)
    y_train = pd.concat([y_train, y_val], axis=0)

    X_train_tfidf = pd.concat([X_train_tfidf, X_val_tfidf], axis=0)
    X_train_pos = pd.concat([X_train_pos, X_val_pos], axis=0)
    X_train_ner = pd.concat([X_train_ner, X_val_ner], axis=0)

    ridge_all_features = ridge(X_train[feat], y_train)
    print('all features accuracy: ', ridge_all_features[0])
    print('all features precision: ', ridge_all_features[1])
    print('all features recall: ', ridge_all_features[2])
    print()

    ridge_text_accuracy = ridge(X_train_tfidf, y_train)
    print('text accuracy: ', ridge_text_accuracy[0])
    print('text precision: ', ridge_text_accuracy[1])
    print('text recall: ', ridge_text_accuracy[2])
    print()

    ridge_pos = ridge(X_train_pos, y_train)
    print('pos accuracy: ', ridge_pos[0])
    print('pos precision: ', ridge_pos[1])
    print('pos recall: ', ridge_pos[2])
    print()

    ridge_ner = ridge(X_train_ner, y_train)
    print('ner accuracy: ', ridge_ner[0])
    print('ner precision: ', ridge_ner[1])
    print('ner recall: ', ridge_ner[2])
    print()

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    ridge_all_features_text = ridge(feat_text_train, y_train)
    print('all features with text tf-idf accuracy: ',
          ridge_all_features_text[0])
    print('all features with text tf-idf precision: ',
          ridge_all_features_text[1])
    print('all features with text tf-idf recall: ', ridge_all_features_text[2])
    print()

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    ridge_all_features_pos = ridge(feat_pos_train, y_train)
    print('all features with pos tf-idf accuracy: ', ridge_all_features_pos[0])
    print('all features with pos tf-idf precision: ',
          ridge_all_features_pos[1])
    print('all features with pos tf-idf recall: ', ridge_all_features_pos[2])
    print()

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    ridge_all_features_ner = ridge(feat_ner_train, y_train)
    print('all features with ner tf-idf accuracy: ', ridge_all_features_ner[0])
    print('all features with ner tf-idf precision: ',
          ridge_all_features_ner[1])
    print('all features with ner tf-idf recall: ', ridge_all_features_ner[2])
    print()

    whole_train = pd.concat([X_train[feat], X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    ridge_whole = ridge(whole_train, y_train)
    print('whole model accuracy: ', ridge_whole[0])
    print('whole model precision: ', ridge_whole[1])
    print('whole model recall: ', ridge_whole[2])
    print()

    top_feat = np.load('top_features.npz')['arr_0']
    condensed_train = whole_train[top_feat]
    ridge_condensed = ridge(condensed_train, y_train)
    print('condensed model accuracy: ', ridge_condensed[0])
    print('condensed model precision: ', ridge_condensed[1])
    print('condensed model recall: ', ridge_condensed[2])
    print()


def ridge(X_train, y_train):
    # Ridge Logistic Regression

    X = np.array(X_train)
    y = np.array(y_train).ravel()

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model = RidgeClassifier(alpha=1.25)
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
