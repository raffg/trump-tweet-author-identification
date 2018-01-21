import pandas as pd
import numpy as np
from src.ridge_grid_scan import ridge_grid_scan
from sklearn.linear_model import RidgeClassifier
from logistic_regression import lr


def main():
    X_train = pd.read_pickle('pickle/train_all_std.pkl')
    y_train = pd.read_pickle('pickle/y_train_all_std.pkl')

    drop = ['created_at', 'id_str', 'in_reply_to_user_id_str', 'tweetokenize',
            'text', 'pos', 'ner']

    # Remove non-numeric features
    X_train = X_train.drop(drop, axis=1)

    # Run feature selection grid scan
    feature_list = ridge_grid_scan(X_train,
                                   np.array(y_train).ravel(),
                                   n=len(X_train.columns))

    print(feature_list)

    feature_list = [(x[0]) for x in list(feature_list)]

    # Save full, sorted feature list
    np.savez('pickle/top_features.npz', feature_list)

    # Save feature list with coefficients
    model = lr(np.array(X_train),
               np.array(y_train).ravel())

    coef = model[3][0]
    features_coefs = list(zip(feature_list, coef))
    np.savez('pickle/features_coefs.npz', features_coefs)


if __name__ == '__main__':
    main()
