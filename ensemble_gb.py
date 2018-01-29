import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    X_train = np.load('pickle/ensemble_predictions_X_train.npz')['arr_0']
    y_train = np.load('pickle/ensemble_predictions_y_train.npz')['arr_0']
    X_test = np.load('pickle/ensemble_predictions_X_test.npz')['arr_0']
    y_test = np.load('pickle/ensemble_predictions_y_test.npz')['arr_0']

    result = gradient_boosting_grid_search(X_train, y_train)
    print(result.best_params_, result.best_score_)

    # model = run_model_gradient_boosting(X_train, y_train)
    # ensemble_save_pickle(model)

    # test_results = ensemble_test_results(model, X_test, y_test)


def run_model_gradient_boosting(X, y):
    ensemble = gradient_boosting(np.array(X), np.array(y).ravel())

    # ensemble_save_pickle(ensemble)


def gradient_boosting(X, y):
    # Basic gradient boosting for ensemble

    kfold = KFold(n_splits=10)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kfold.split(X):
        model = GradientBoostingClassifier()
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


def gradient_boosting_grid_search(X, y):
    parameters = {'loss': ['deviance', 'exponential'],
                  'learning_rate': [.01, .05, .1, .5, 1],
                  'n_estimators': [75, 100, 150, 200],
                  'max_depth': [2, 3, 4, 5, 6],
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2, 3],
                  'min_weight_fraction_leaf': [0., .01, .1, .5],
                  'subsample': [.25, .5, .75, 1],
                  'max_features': [None, 'sqrt', 'log2']}

    gb = GradientBoostingClassifier()
    clf = GridSearchCV(gb, parameters, cv=3, verbose=True)
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
    output = open('pickle/ensemble_gb.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
