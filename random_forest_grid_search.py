import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def main():
    # Load the data
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle('pickle/data.pkl')

    # Performing cross-validation, don't need to separate train and validation
    (X_train, X_train_tfidf, X_train_pos, X_train_ner,
     X_test, X_test_tfidf, X_test_pos, X_test_ner) = cross_val_data(X_train,
                                                                    X_val,
                                                                    X_test)
    # Concatenate all training DataFrames
    X_train = pd.concat([X_train, X_train_tfidf,
                         X_train_pos, X_train_ner], axis=1)
    X_test = pd.concat([X_test, X_test_tfidf,
                        X_test_pos, X_test_ner], axis=1)
    y_train = pd.concat([y_train, y_val], axis=0)

    feat = np.load('pickle/top_features.npz')['arr_0']

    results = []
    for n in np.arange(100, len(feat) + 1, 100):
        result = random_forest_grid_search(np.array(X_train[feat[:n]]),
                                           np.array(y_train).ravel())
        results.append((n, result.best_params_, result.best_score_))
        print(n, result.best_params_, result.best_score_)

    for item in results:
        print(item[0], item[1], item[2])


def random_forest_grid_search(X, y):
    parameters = {'n_estimators': [8, 10, 15, 20, 30],
                  'max_features': [None, 'sqrt', 'log2'],
                  'max_depth': [None, 3, 5, 10, 20],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1, 2, 5],
                  'max_leaf_nodes': [10, 25, 50, 100, None],
                  'n_jobs': [-1]}

    '''
    Results:
    {'max_depth': 20, 'max_features': None, 'max_leaf_nodes': None,
    'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 30,
    'n_jobs': -1} 0.943672304871
    '''

    parameters2 = {'n_estimators': [25, 30],
                   'max_features': [None, 'sqrt', 'log2'],
                   'max_depth': [20, 30],
                   'min_samples_split': [2, 3],
                   'min_samples_leaf': [2, 3],
                   'n_jobs': [-1]}

    '''
    Results (150 features):
    {'max_depth': 20, 'max_features': None, 'min_samples_leaf': 2,
    'min_samples_split': 2, 'n_estimators': 30, 'n_jobs': -1} 0.94122681883
    '''

    parameters3 = {'n_estimators': [25, 30],
                   'max_depth': [20, 30],
                   'min_samples_split': [2],
                   'min_samples_leaf': [1, 2],
                   'n_jobs': [-1]}

    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters3, cv=5, verbose=True)
    clf.fit(X, y)

    return clf


if __name__ == '__main__':
    main()
