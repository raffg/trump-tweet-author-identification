import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from sklearn.naive_bayes import MultinomialNB
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
    print(len(feat))

    results = []
    for n in range(1, 20):
        result = naive_bayes_grid_search(np.array(X_train[feat[0:n]]),
                                         np.array(y_train).ravel())
        results.append((n, result.best_params_, result.best_score_))
        print(n, result.best_params_, result.best_score_)

    for item in results:
        print(item[0], item[1], item[2])


def naive_bayes_grid_search(X, y):
    parameters = {'alpha': [1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100,
                            500, 750, 875, 1000, 1125, 1250, 1375, 1500, 2000,
                            2500, 3000, 10000, 100000]}

    nb = MultinomialNB()
    clf = GridSearchCV(nb, parameters, verbose=True)
    clf.fit(X, y)

    return clf


if __name__ == '__main__':
    main()
