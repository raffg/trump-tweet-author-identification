import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_lasso_regression('pickle/data.pkl')


def run_model_lasso_regression(file):
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

    lasso_all_features = lasso(X_train[feat], y_train)
    print('all features accuracy: ', lasso_all_features[0])
    print('all features precision: ', lasso_all_features[1])
    print('all features recall: ', lasso_all_features[2])
    print()

    lasso_text_accuracy = lasso(X_train_tfidf, y_train)
    print('text accuracy: ', lasso_text_accuracy[0])
    print('text precision: ', lasso_text_accuracy[1])
    print('text recall: ', lasso_text_accuracy[2])
    print()

    lasso_pos = lasso(X_train_pos, y_train)
    print('pos accuracy: ', lasso_pos[0])
    print('pos precision: ', lasso_pos[1])
    print('pos recall: ', lasso_pos[2])
    print()

    lasso_ner = lasso(X_train_ner, y_train)
    print('ner accuracy: ', lasso_ner[0])
    print('ner precision: ', lasso_ner[1])
    print('ner recall: ', lasso_ner[2])
    print()

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    lasso_all_features_text = lasso(feat_text_train, y_train)
    print('all features with text tf-idf accuracy: ',
          lasso_all_features_text[0])
    print('all features with text tf-idf precision: ',
          lasso_all_features_text[1])
    print('all features with text tf-idf recall: ', lasso_all_features_text[2])
    print()

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    lasso_all_features_pos = lasso(feat_pos_train, y_train)
    print('all features with pos tf-idf accuracy: ', lasso_all_features_pos[0])
    print('all features with pos tf-idf precision: ',
          lasso_all_features_pos[1])
    print('all features with pos tf-idf recall: ', lasso_all_features_pos[2])
    print()

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    lasso_all_features_ner = lasso(feat_ner_train, y_train)
    print('all features with ner tf-idf accuracy: ', lasso_all_features_ner[0])
    print('all features with ner tf-idf precision: ',
          lasso_all_features_ner[1])
    print('all features with ner tf-idf recall: ', lasso_all_features_ner[2])
    print()

    whole_train = pd.concat([X_train[feat], X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    lasso_whole = lasso(whole_train, y_train)
    print('whole model accuracy: ', lasso_whole[0])
    print('whole model precision: ', lasso_whole[1])
    print('whole model recall: ', lasso_whole[2])
    print()

    top_feat = np.load('pickle/top_features.npz')['arr_0'][:100]
    condensed_train = whole_train[top_feat]
    lasso_condensed = lasso(condensed_train, y_train)
    print('condensed model accuracy: ', lasso_condensed[0])
    print('condensed model precision: ', lasso_condensed[1])
    print('condensed model recall: ', lasso_condensed[2])
    print()


def lasso(X_train, y_train):
    # Lasso Logistic Regression

    X = np.array(X_train)
    y = np.array(y_train).ravel()

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model = Lasso()
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
