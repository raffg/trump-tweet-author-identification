import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_logistic_regression('data/data.pkl')


def run_model_logistic_regression(file):
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     y_train, y_val, y_test) = load_pickle(file)

    binary = ['is_retweet', 'is_reply', 'is_quoted_retweet',
              'tweetstorm', 'period_1', 'period_2', 'period_3',
              'period_4']

    continuous = ['favorite_count', 'retweet_count', 'compound', 'negative',
                  'neutral', 'positive', 'tweet_length',
                  'avg_sentence_length', 'avg_word_length', 'commas',
                  'semicolons', 'exclamations', 'periods', 'questions',
                  'quotes', 'mentions', 'hashtags', 'urls', 'all_caps',
                  'period_1', 'period_2', 'period_3', 'period_4']

    feat = continuous

    X_train = pd.concat([X_train, X_val], axis=0)
    y_train = pd.concat([y_train, y_val], axis=0)

    X_train_tfidf = pd.concat([X_train_tfidf, X_val_tfidf], axis=0)
    X_train_pos = pd.concat([X_train_pos, X_val_pos], axis=0)

    ridge_all_features = ridge(np.array(X_train[feat]),
                               np.array(y_train).ravel())
    print('all features accuracy: ', ridge_all_features[0])
    print('all features precision: ', ridge_all_features[1])
    print('all features recall: ', ridge_all_features[2])
    print()

    ridge_text_accuracy = ridge(np.array(X_train_tfidf),
                                np.array(y_train).ravel())
    print('text accuracy: ', ridge_text_accuracy[0])
    print('text precision: ', ridge_text_accuracy[1])
    print('text recall: ', ridge_text_accuracy[2])
    print()

    ridge_pos_n_grams = ridge(np.array(X_train_pos), np.array(y_train).ravel())
    print('pos accuracy: ', ridge_pos_n_grams[0])
    print('pos precision: ', ridge_pos_n_grams[1])
    print('pos recall: ', ridge_pos_n_grams[2])
    print()

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    ridge_all_features_text = ridge(np.array(feat_text_train),
                                    np.array(y_train).ravel())
    print('all features with text tf-idf accuracy: ',
          ridge_all_features_text[0])
    print('all features with text tf-idf precision: ',
          ridge_all_features_text[1])
    print('all features with text tf-idf recall: ', ridge_all_features_text[2])
    print()

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    ridge_all_features_pos = ridge(np.array(feat_pos_train),
                                   np.array(y_train).ravel())
    print('all features with pos tf-idf accuracy: ', ridge_all_features_pos[0])
    print('all features with pos tf-idf precision: ',
          ridge_all_features_pos[1])
    print('all features with pos tf-idf recall: ', ridge_all_features_pos[2])
    print()

    vader = ridge(np.array(X_train[['compound', 'negative',
                                    'neutral', 'positive']]),
                  np.array(y_train).ravel())
    print('vader features accuracy: ', vader[0])
    print('vader features precision: ', vader[1])
    print('vader features recall: ', vader[2])
    print()


def ridge(X_train, y_train):
    # Ridge Logistic Regression

    X = np.array(X_train)
    y = np.array(y_train)

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model = Ridge()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test).round()
        y_true = y_test
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict,
                          average='weighted'))
        recalls.append(recall_score(y_true, y_predict, average='weighted'))

    return np.average(accuracies), np.average(precisions), np.average(recalls)


if __name__ == '__main__':
    main()
