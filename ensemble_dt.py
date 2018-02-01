import pandas as pd
import numpy as np
import pickle
from ensemble import save_pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    # Load the data
    pkl = open('pickle/ensemble_results.pkl', 'rb')
    rf_results = pickle.load(pkl)
    ab_results = pickle.load(pkl)
    gb_results = pickle.load(pkl)
    knn_results = pickle.load(pkl)
    nb_results = pickle.load(pkl)
    gnb_results = pickle.load(pkl)
    svc_results = pickle.load(pkl)
    svm_results = pickle.load(pkl)
    lr_results = pickle.load(pkl)
    y_train = pickle.load(pkl)
    pkl.close()

    data = {'rf': rf_results, 'ab': ab_results, 'gb': gb_results,
            'knn': knn_results, 'nb': nb_results, 'gnb': gnb_results,
            'svc': svc_results, 'svm': svm_results, 'lr': lr_results}

    X = pd.DataFrame(data)
    y = y_train

    X['majority'] = X.apply(majority, axis=1)

    # Split test and train data
    (X_train, X_test, y_train, y_test) = train_test_split(X, y,
                                                          test_size=0.2,
                                                          random_state=1)

    # result = decision_tree_grid_search(X, y)
    # print(result.best_params_, result.best_score_)

    ensemble = run_model_decision_tree(X_train, y_train)
    # save_pickle([ensemble], 'pickle/ensemble_decision_tree.pkl')

    test_results = ensemble_test_results(ensemble, X_test, y_test)


def run_model_decision_tree(X, y):
    model = decision_tree(np.array(X), np.array(y).ravel())
    return model


def decision_tree(X, y):
    # Basic decision tree for ensemble

    kfold = KFold(n_splits=10)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kfold.split(X):
        model = DecisionTreeClassifier(criterion='gini',
                                       max_depth=None,
                                       min_weight_fraction_leaf=0.001,
                                       splitter='best')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_true = y_test
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))
        f1_scores.append(f1_score(y_true, y_predict))

    accuracy = np.average(accuracies)
    precision = np.average(precisions)
    recall = np.average(recalls)
    f1 = np.average(f1_scores)

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', f1)

    return model


def decision_tree_grid_search(X, y):
    parameters = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'max_depth': [None, 2, 3, 4, 5, 6],
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2, 3],
                  'min_weight_fraction_leaf': [0., .001, .01, .1, .25],
                  'max_features': [None, 'sqrt', 'log2']}

    dt = DecisionTreeClassifier()
    clf = GridSearchCV(dt, parameters, cv=10, verbose=True)
    clf.fit(X, y)

    return clf


def majority(row):
    val = 1 if (row['rf'] + row['ab'] + row['gb'] + row['knn'] + row['nb'] +
                row['gnb'] + row['svc'] + row['svm'] + row['lr']) > 3 else 0
    return val


def ensemble_test_results(model, X_test, y_test):
    y_predict = model.predict(X_test)

    print()
    print('Accuracy: ', accuracy_score(y_test, y_predict))
    print('Precision: ', precision_score(y_test, y_predict))
    print('Recall: ', recall_score(y_test, y_predict))
    print('F1 score: ', f1_score(y_test, y_predict))


if __name__ == '__main__':
    main()
