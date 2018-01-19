import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_logistic_regression('pickle/data.pkl')


def run_model_logistic_regression(file):
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle(file)

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    X_train = pd.concat([X_train, X_val], axis=0)
    (X_train, X_test) = standardize(feat, X_train, X_test)
    y_train = pd.concat([y_train, y_val], axis=0)

    X_train_tfidf = pd.concat([X_train_tfidf, X_val_tfidf], axis=0)
    X_train_pos = pd.concat([X_train_pos, X_val_pos], axis=0)
    X_train_ner = pd.concat([X_train_ner, X_val_ner], axis=0)

    lr_all_features = lr(np.array(X_train[feat]), np.array(y_train).ravel())
    print('all features accuracy: ', lr_all_features[0])
    print('all features precision: ', lr_all_features[1])
    print('all features recall: ', lr_all_features[2])
    print()

    lr_text_accuracy = lr(np.array(X_train_tfidf), np.array(y_train).ravel())
    print('text accuracy: ', lr_text_accuracy[0])
    print('text precision: ', lr_text_accuracy[1])
    print('text recall: ', lr_text_accuracy[2])
    print()

    lr_pos = lr(np.array(X_train_pos), np.array(y_train).ravel())
    print('pos accuracy: ', lr_pos[0])
    print('pos precision: ', lr_pos[1])
    print('pos recall: ', lr_pos[2])
    print()

    lr_ner = lr(np.array(X_train_ner), np.array(y_train).ravel())
    print('ner accuracy: ', lr_ner[0])
    print('ner precision: ', lr_ner[1])
    print('ner recall: ', lr_ner[2])
    print()

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    lr_all_features_text = lr(np.array(feat_text_train),
                              np.array(y_train).ravel())
    print('all features with text tf-idf accuracy: ', lr_all_features_text[0])
    print('all features with text tf-idf precision: ', lr_all_features_text[1])
    print('all features with text tf-idf recall: ', lr_all_features_text[2])
    print()

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    lr_all_features_pos = lr(np.array(feat_pos_train),
                             np.array(y_train).ravel())
    print('all features with pos tf-idf accuracy: ', lr_all_features_pos[0])
    print('all features with pos tf-idf precision: ', lr_all_features_pos[1])
    print('all features with pos tf-idf recall: ', lr_all_features_pos[2])
    print()

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    lr_all_features_ner = lr(np.array(feat_ner_train),
                             np.array(y_train).ravel())
    print('all features with ner tf-idf accuracy: ', lr_all_features_ner[0])
    print('all features with ner tf-idf precision: ', lr_all_features_ner[1])
    print('all features with ner tf-idf recall: ', lr_all_features_ner[2])
    print()

    whole_train = pd.concat([X_train[feat], X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    lr_whole = lr(np.array(whole_train),
                  np.array(y_train).ravel())
    print('whole model accuracy: ', lr_whole[0])
    print('whole model precision: ', lr_whole[1])
    print('whole model recall: ', lr_whole[2])
    print()

    top_feat = np.load('pickle/top_features.npz')['arr_0'][:20]
    condensed_train = whole_train[top_feat]
    lr_condensed = lr(np.array(condensed_train),
                      np.array(y_train).ravel())
    print('condensed model accuracy: ', lr_condensed[0])
    print('condensed model precision: ', lr_condensed[1])
    print('condensed model recall: ', lr_condensed[2])
    print()


def lr(X_train, y_train):
    # Basic Logistic Regression

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


if __name__ == '__main__':
    main()
