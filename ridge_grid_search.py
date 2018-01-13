import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    scores = grid_search('pickle/data.pkl')
    plot_alphas(scores)


def grid_search(file):
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

    scores = []
    n_alphas = 500
    alphas = np.logspace(.75, 3, n_alphas)
    highest_score = 0

    for a in alphas:
        score = ridge(X_train[feat], y_train, a)
        scores.append((score[0], a))
        if score[0] > highest_score:
            highest_score = score[0]
            highest_alpha = a
        print('accuracy: ', score[0], ',   alpha: ', a)
    print()
    print('highest accuracy of ', highest_score,
          'at alpha of ', highest_alpha)

    return scores


def plot_alphas(alphas):
    '''
    Takes a list of n alphas and plots the accuracy as a function of number of
    alphas
    INPUT: list
    OUTPUT:
    '''

    ax = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    ax.plot([item[1] for item in alphas], [item[0] for item in alphas])
    ax.set_xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Ridge accuracies as a function of alpha')
    plt.axis('tight')
    plt.show()


def ridge(X_train, y_train, alpha):
    # Ridge Logistic Regression

    X = np.array(X_train)
    y = np.array(y_train).ravel()

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model = RidgeClassifier(alpha=alpha, fit_intercept=False)
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
