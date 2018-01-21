import pandas as pd
import numpy as np
from src.ridge_grid_scan import ridge_grid_scan
from sklearn.linear_model import RidgeClassifier
from logistic_regression import lr

from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from src.standardize import standardize
from sklearn.model_selection import train_test_split


def main():
    '''
    X_train = pd.read_pickle('pickle/train_all_std.pkl')
    y_train = pd.read_pickle('pickle/y_train_all_std.pkl')

    drop = ['created_at', 'id_str', 'in_reply_to_user_id_str', 'tweetokenize',
            'text', 'pos', 'ner']

    # Remove non-numeric features
    X_train = X_train.drop(drop, axis=1)
    '''
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle('pickle/old_pickle/data.pkl')

    # Recomine all X data, apply date mask, recreate train/test splits
    X = pd.concat([X_train, X_val, X_test], axis=0).reset_index(drop=True)
    mask = (X['created_at'] < '2017-03-26')
    X = X.loc[mask]
    y = pd.DataFrame(np.where(X['source'] == 'Twitter for Android', 1, 0))
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
    (X_train, X_val, y_train,
     y_val) = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    (X_train, X_train_tfidf, X_train_pos, X_train_ner,
     X_test, X_test_tfidf, X_test_pos, X_test_ner) = cross_val_data(X_train,
                                                                    X_val,
                                                                    X_test)
    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08']
    (X_train, X_test) = standardize(feat, X_train, X_test)
    X_train = pd.concat([X_train, X_train_tfidf,
                         X_train_pos, X_train_ner], axis=1)
    X_test = pd.concat([X_test, X_test_tfidf,
                        X_test_pos, X_test_ner], axis=1)
    y_train = pd.concat([y_train, y_val], axis=0)

    X_train.to_pickle('pickle/X_train_fixed.pkl')
    X_test.to_pickle('pickle/X_test_fixed.pkl')
    y_train.to_pickle('pickle/y_train_fixed.pkl')
    y_test.to_pickle('pickle/y_test_fixed.pkl')



    # Run feature selection iterations
    feature_list = ridge_grid_scan(X_train,
                                   np.array(y_train).ravel(),
                                   n=len(X_train.columns))

    print(feature_list)

    feature_list = [(x[0]) for x in list(feature_list)]

    # Save full, sorted feature list
    np.savez('pickle/top_features_testing.npz', feature_list)

    # Save feature list with coefficients
    model = lr(np.array(X_train),
               np.array(y_train).ravel())

    coef = model[3][0]
    features_coefs = list(zip(feature_list, coef))
    np.savez('pickle/features_coefs_testing.npz', features_coefs)


if __name__ == '__main__':
    main()
