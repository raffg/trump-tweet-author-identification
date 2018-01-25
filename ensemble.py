import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    run_model_random_forest()


def run_model_random_forest():
    X = np.load('pickle/ensemble_predictions_X.npz')['arr_0']
    y = np.load('pickle/ensemble_predictions_y.npz')['arr_0']

    ensemble = random_forest(np.array(X), np.array(y).ravel())

    # ensemble_save_pickle(ensemble)


def random_forest(X, y):
    # Basic random forest for ensemble

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kfold.split(X):
        model = RandomForestClassifier()
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


def random_forest_save_pickle(model):
    # Save pickle file
    output = open('pickle/ensemble.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
