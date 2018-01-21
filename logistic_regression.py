import pandas as pd
import numpy as np
import pickle
from src.load_pickle import load_pickle
from src.standardize import standardize
from src.cross_val_data import cross_val_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_logistic_regression()
    # lr_save_pickle()


def run_model_logistic_regression():
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

    lr_all_features = lr(np.array(X_train[feat]), np.array(y_train).ravel())
    print('all features accuracy: ', lr_all_features[0])
    print('all features precision: ', lr_all_features[1])
    print('all features recall: ', lr_all_features[2])
    print()

    whole_train = X_train
    lr_whole = lr(np.array(whole_train),
                  np.array(y_train).ravel())
    print('whole model accuracy: ', lr_whole[0])
    print('whole model precision: ', lr_whole[1])
    print('whole model recall: ', lr_whole[2])
    print()

    # top_feat = np.load('pickle/top_features.npz')['arr_0'][:20]
    # condensed_train = whole_train[top_feat]
    # lr_condensed = lr(np.array(condensed_train),
    #                   np.array(y_train).ravel())
    # print('condensed model accuracy: ', lr_condensed[0])
    # print('condensed model precision: ', lr_condensed[1])
    # print('condensed model recall: ', lr_condensed[2])
    # print()


def lr(X_train, y_train):
    # Cross-validated Logistic Regression

    X = np.array(X_train)
    y = np.array(y_train)

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model = LogisticRegression(C=.05)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_true = y_test
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    return (np.average(accuracies), np.average(precisions),
            np.average(recalls), model.coef_)


def lr_save_pickle():
    # Basic Logistic Regression, save pickle

    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle('pickle/data.pkl')

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    (X_train, X_train_tfidf, X_train_pos, X_train_ner,
     X_test, X_test_tfidf, X_test_pos, X_test_ner) = cross_val_data(X_train,
                                                                    X_val,
                                                                    X_test)
    (X_train, X_test) = standardize(feat, X_train, X_test)

    # Concatenate all training DataFrames
    X_train = pd.concat([X_train, X_train_tfidf,
                         X_train_pos, X_train_ner], axis=1)
    X_test = pd.concat([X_test, X_test_tfidf,
                        X_test_pos, X_test_ner], axis=1)
    y_train = pd.concat([y_train, y_val], axis=0)

    X = np.array(X_train)
    y = np.array(y_train).ravel()

    accuracies = []
    precisions = []
    recalls = []

    lr = LogisticRegression(C=.05)
    lr.fit(X, y)

    # Save pickle file
    output = open('pickle/logistic_regression_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(lr, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
