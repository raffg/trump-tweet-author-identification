import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    X_train = np.load('pickle/ensemble_predictions_X_train.npz')['arr_0']
    y_train = np.load('pickle/ensemble_predictions_y_train.npz')['arr_0']
    X_test = np.load('pickle/ensemble_predictions_X_test.npz')['arr_0']
    y_test = np.load('pickle/ensemble_predictions_y_test.npz')['arr_0']

    result = random_forest_grid_search(X_train, y_train)
    print(result.best_params_, result.best_score_)

    # model = run_model_random_forest(X_train, y_train)
    # ensemble_save_pickle(model)

    # test_results = ensemble_test_results(model, X_test, y_test)


def run_model_random_forest(X, y):
    ensemble = random_forest(np.array(X), np.array(y).ravel())

    # ensemble_save_pickle(ensemble)


def random_forest(X, y):
    # Basic random forest for ensemble

    kfold = KFold(n_splits=10)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kfold.split(X):
        model = RandomForestClassifier(max_depth=None,
                                       max_features='sqrt',
                                       max_leaf_nodes=None,
                                       min_samples_leaf=2,
                                       min_samples_split=7,
                                       n_estimators=1000,
                                       n_jobs=-1)
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


def random_forest_grid_search(X, y):
    parameters = {'n_estimators': [1000],
                  'max_features': ['sqrt', 'log2'],
                  'max_depth': [None],
                  'min_samples_split': [5, 6, 7, 8, 10],
                  'min_samples_leaf': [1, 2, 3],
                  'max_leaf_nodes': [None],
                  'n_jobs': [-1]}

    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, cv=10, verbose=True)
    clf.fit(X, y)

    return clf


def ensemble_test_results(model, X_test, y_test):
    y_predict = model.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, y_predict))
    print('Precision: ', precision_score(y_test, y_predict))
    print('Recall: ', recall_score(y_test, y_predict))
    print('F1 score: ', f1_score(y_test, y_predict))


def ensemble_save_pickle(model):
    # Save pickle file
    output = open('pickle/ensemble_rf.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
