import pandas as pd
import numpy as np
import operator
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
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
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

    whole_train = pd.concat([X_train[feat], X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    ridge_whole = ridge(whole_train, y_train)
    print('    Whole model accuracy: ', ridge_whole[0])

    feat_coef = dict(zip(whole_train.columns, ridge_whole[3][0]))
    sorted_list = sorted(feat_coef.items(), key=operator.itemgetter(1))

    top_n = 283
    top_trump = sorted_list[::-1][:top_n]
    top_not_trump = sorted_list[:top_n]

    trump_feat = [item[0] for item in top_trump]
    not_trump_feat = [item[0] for item in top_not_trump]

    top_feat = trump_feat + not_trump_feat

    top_ridge = ridge(whole_train[top_feat], y_train)
    print('Condensed model accuracy: ', top_ridge[0])

    np.savez('top_features.npz', top_feat)

    '''
    best_n = 0
    best_n_reduced = 0
    best_n_reduced2 = 0
    accuracy = 0
    accuracy_reduced = 0
    accuracy_reduced2 = 0
    for n in range(1, 500):
        top_n = n
        top_trump = sorted_list[::-1][:top_n]
        top_not_trump = sorted_list[:top_n]

        trump_feat = [item[0] for item in top_trump]
        not_trump_feat = [item[0] for item in top_not_trump]

        top_feat = trump_feat + not_trump_feat

        top_ridge = ridge(whole_train[top_feat], y_train)

        if top_ridge[0] > accuracy:
            accuracy = top_ridge[0]
            best_n = n

        if top_ridge[0] > accuracy_reduced + .005:
            accuracy_reduced = top_ridge[0]
            best_n_reduced = n

        if top_ridge[0] > accuracy_reduced2 + .01:
            accuracy_reduced2 = top_ridge[0]
            best_n_reduced2 = n

        print(n, 'n model accuracy: ', top_ridge[0])
    print(best_n, 'n model accuracy: ', accuracy)
    print(best_n_reduced, 'n model accuracy: ', accuracy_reduced)
    print(best_n_reduced2, 'n model accuracy: ', accuracy_reduced2)
    '''

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
