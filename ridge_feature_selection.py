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
            'all_caps', 'tweetstorm', 'hour', 'period_1', 'period_2',
            'period_3', 'period_4']

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
    feature_list = ridge_grid_scan(X_train, y_train)

    print(feature_list)

    # Save full, sorted feature list
    np.savez('top_features.npz', feature_list)

    # Plot accuracies
    # (accuracies, top_accuracies) = ridge_feature_iteration(whole_train,
    #                                                       y_train,
    #                                                       feature_list)

    # plot_accuracies(accuracies, 'Ridge')

    # Save feature list with highest accuracy
    # save_top_feature_list(top_accuracies[0][0], feature_list)


def ridge_feature_selection(X, y):
    '''
    Takes a feature DataFrame and a label DataFrame and iterates through alpha
    values of ridge regression to drop out features creates an ordered feature
    list.
    INPUT: DataFrame, DataFrame
    OUTPUT: list
    '''
    features = []
    for alpha in np.geomspace(1e-1, 1e8, 200):
        print('alpha: ', alpha)
        model = ridge(X, y, alpha)
        feat_coef = list(zip(X.columns, model[3][0]))
        for feature in feat_coef:
            if (abs(feature[1]) < 1e-3 and feature[0] not in features):
                features.append(feature[0])
        print(len(features))
        print('% complete: ', 100 * (len(features) / len(feat_coef)))
        print()
        if not (set(X.columns) - set(features)):
            break
    return list(reversed(features))


def create_feature_list(X, y):
    '''
    Creates a list of features, sorted by importance
    INPUT: DataFrame of tweet data and DataFrame of labels
    OUTPUT: list of tuples of the feature name and its ridge importance score
    '''

    ridge_whole = ridge(X, y)

    feat_coef = dict(zip(X.columns, ridge_whole[3][0]))
    sorted_list = sorted(feat_coef.items(), key=operator.itemgetter(1))

    np.savez('sorted_data_pos_corrected_ner.npz', sorted_list)

    abs_sorted_list = [(item[0], abs(item[1])) for item in sorted_list]
    abs_sorted_list = sorted(abs_sorted_list, key=lambda x: x[1])[::-1]

    return abs_sorted_list


def ridge_feature_iteration(X, y, feature_list):
    '''
    Takes a DataFrame and a sorted feature list and iteratively performs ridge
    regression on an increasing number of features to produce a list of
    accuracy scores and the number of features corresponding to the highest
    accuracy, the highest accuracy minus .5%, and the highest accuracy minus 1%
    INPUT: X DataFrame, y DataFrame, list of features
    OUTPUT: accuracy list, list of tuples
    '''
    best_n = 0
    best_n_reduced = 0
    best_n_reduced2 = 0

    accuracy = 0
    accuracy_reduced = 0
    accuracy_reduced2 = 0

    accuracies = []

    for n in range(1, len(feature_list)):
        top_feat = [item[0] for item in feature_list[:n]]
        top_ridge = ridge(X[top_feat], y)

        accuracies.append(top_ridge[0])

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
    print()
    print(best_n, 'feature model accuracy: ', accuracy)
    print(best_n_reduced, 'feature model accuracy: ', accuracy_reduced)
    print(best_n_reduced2, 'feature model accuracy: ', accuracy_reduced2)

    top_accuracies = [(best_n, accuracy),
                      (best_n_reduced, accuracy_reduced),
                      (best_n_reduced2, accuracy_reduced2)]

    return accuracies, top_accuracies


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


def save_top_feature_list(number_of_features, feature_list):
    '''
    Takes the number of features to keep and saves the top N features to .npz
    INPUT: int: N number of features to keep
    OUTPUT:
    '''

    top_feat = [item[0] for item in feature_list[:number_of_features]]

    np.savez('top_features.npz', top_feat)


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
