import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
from src.ridge_grid_scan import ridge_grid_scan
from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from src.standardize import standardize
from sklearn.linear_model import RidgeClassifier


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
            'hour_08_14', 'hour_02_08']

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

    # Run feature selection iterations
    feature_list = ridge_grid_scan(X_train,
                                   np.array(y_train).ravel(),
                                   n=len(X_train.columns))

    print(feature_list)

    feature_list = [(x[0]) for x in list(feature_list)]

    # Save full, sorted feature list
    np.savez('pickle/top_features.npz', feature_list)

    # Plot accuracies
    # (accuracies, top_accuracies) = ridge_feature_iteration(whole_train,
    #                                                       y_train,
    #                                                       feature_list)

    # plot_accuracies(accuracies, 'Ridge')


def save_top_feature_list(number_of_features, feature_list, filename):
    '''
    Takes the number of features to keep and saves the top N features to .npz
    INPUT: int: N number of features to keep, feature list, filename string
    OUTPUT:
    '''

    top_feat = [item[0] for item in feature_list[:number_of_features]]

    np.savez(filename, top_feat)


def plot_accuracies(accuracies, model):
    '''
    Takes a list of list of n accuracies from n number of features and plots
    the accuracy as a function of number of features. Model name required for
    plot title.
    INPUT: list, string
    OUTPUT:
    '''

    ax = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    ax.plot(range(len(accuracies)), accuracies)
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title(model + ' accuracies as a function of the number of features')
    plt.axis('tight')
    plt.show()


def ridge(X_train, y_train, alpha=50):
    # Ridge Logistic Regression

    X = np.array(X_train)
    y = np.array(y_train).ravel()

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold.split(X):
        model = RidgeClassifier(alpha=alpha)
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
