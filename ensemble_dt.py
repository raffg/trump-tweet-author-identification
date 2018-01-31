import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
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
    svm_results = pickle.load(pkl)
    lr_results = pickle.load(pkl)
    y_train = pickle.load(pkl)
    pkl.close()

    data = {'rf': rf_results, 'ab': ab_results, 'gb': gb_results,
            'knn': knn_results, 'nb': nb_results, 'gnb': gnb_results,
            'svm': svm_results, 'lr': lr_results}

    X = pd.DataFrame(data)
    y = y_train

    X['majority'] = X.apply(majority, axis=1)

    # result = decision_tree_grid_search(X, y)
    # print(result.best_params_, result.best_score_)

    model = run_model_decision_tree(X, y)
    # ensemble_save_pickle(model)

    # test_results = ensemble_test_results(model, X_test, y_test)


def run_model_decision_tree(X, y):
    ensemble = decision_tree(np.array(X), np.array(y).ravel())

    # ensemble_save_pickle(ensemble)


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

    return accuracy, precision, recall, f1


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
    val = 1 if (row['rf'] + row['ab'] + row['gb'] + row['knn'] + row['gnb'] +
                row['svm'] + row['lr']) > 3 else 0
    return val


def ensemble_test_results(model, X_test, y_test):
    y_predict = model.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, y_predict))
    print('Precision: ', precision_score(y_test, y_predict))
    print('Recall: ', recall_score(y_test, y_predict))
    print('F1 score: ', f1_score(y_test, y_predict))


def ensemble_save_pickle(model):
    # Save pickle file
    output = open('pickle/ensemble_dt.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


def open_pickle(filename):
    with open(filename, 'rb') as pickle:
        return pickle.load(pickle)


if __name__ == '__main__':
    main()
