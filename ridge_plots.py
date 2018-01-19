import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
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

    X_train = pd.concat([X_train, X_val], axis=0)
    (X_train, X_test) = standardize(feat, X_train, X_test)
    y_train = pd.concat([y_train, y_val], axis=0)

    scores = grid_search(X_train[feat], y_train)
    # plot_alphas(scores, 'Ridge')
    plot_regularization(X_train[feat], y_train)


def grid_search(X, y):

    scores = []
    n_alphas = 500
    alphas = np.logspace(.75, 3, n_alphas)
    highest_score = 0

    for a in alphas:
        score = ridge(X, y, a)
        scores.append((score[0], a))
        if score[0] > highest_score:
            highest_score = score[0]
            highest_alpha = a
        print('accuracy: ', score[0], ',   alpha: ', a)
    print()
    print('highest accuracy of ', highest_score,
          'at alpha of ', highest_alpha)

    return scores


def plot_alphas(alphas, model):
    '''
    Takes a list of n alphas and plots the accuracy as a function of number of
    alphas. Model name required for title.
    INPUT: list, string
    OUTPUT:
    '''

    ax = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    ax.plot([item[1] for item in alphas], [item[0] for item in alphas])
    ax.set_xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title(model + ' accuracies as a function of alpha')
    plt.axis('tight')
    plt.show()


def plot_regularization(X, y):
    # #############################################################################
    # Compute paths

    n_alphas = 200
    alphas = np.logspace(1, 7, n_alphas)

    coefs = []
    for a in alphas:
        ridge = RidgeClassifier(alpha=a, fit_intercept=False)
        ridge.fit(X, np.array(y).ravel())
        coefs.append(ridge.coef_[0])

    # #############################################################################
    # Display results

    ax = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
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
