import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from src.standardize import standardize
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV


def main():
    # Load the data
    X_train = pd.read_pickle('pickle/train_val_all_std.pkl')
    X_val = pd.read_pickle('pickle/val_all_std.pkl')
    y_train = pd.read_pickle('pickle/y_train_val_all_std.pkl')
    y_val = pd.read_pickle('pickle/y_val_all_std.pkl')

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    drop = ['created_at', 'id_str', 'in_reply_to_user_id_str', 'tweetokenize',
            'text', 'pos', 'ner']

    X_train = X_train.drop(drop, axis=1)

    result = ridge_grid_search(np.array(X_train),
                               np.array(y_train).ravel())
    print(result.best_params_, result.best_score_)


def ridge_grid_search(X, y):
    parameters = {'alpha': [1e-5, 1e-3, 1e-1, 1, 10, 100, 1000, 10000]}

    ridge = RidgeClassifier()
    clf = GridSearchCV(ridge, parameters, cv=5, verbose=True)
    clf.fit(X, y)

    return clf


if __name__ == '__main__':
    main()
