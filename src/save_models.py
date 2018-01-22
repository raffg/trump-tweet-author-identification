import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier


def main():
    run_model_random_forest()
    run_model_ridge_regression()


def run_model_random_forest():
    X_train = pd.read_pickle('pickle/X_labeled.pkl')
    y_train = pd.read_pickle('pickle/y.pkl')

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

    top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:50])
    X_feat = []
    for feat in top_feat:
        if feat in X_train.columns:
            X_feat.append(feat)

    print('Running random forest')
    condensed_train = X_train[X_feat]
    random_forest_condensed = random_forest(np.array(condensed_train),
                                            np.array(y_train).ravel())
    random_forest_save_pickle(random_forest_condensed)


def random_forest(X_train, y_train):
    # Basic random forest
    rf = RandomForestClassifier(max_depth=20,
                                max_features='sqrt',
                                max_leaf_nodes=None,
                                min_samples_leaf=2,
                                min_samples_split=2,
                                n_estimators=1000,
                                n_jobs=-1).fit(X_train, y_train)

    return rf


def random_forest_save_pickle(model):
    # Save pickle file
    output = open('pickle/random_forest_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


def run_model_ridge_regression():
    X_train = pd.read_pickle('pickle/X_labeled_std.pkl')
    y_train = pd.read_pickle('pickle/y.pkl')

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

    print('Running ridge regression')
    ridge_model = ridge(np.array(X_train), np.array(y_train).ravel())
    ridge_save_pickle(ridge_model)


def ridge(X_train, y_train):
    # Ridge Logistic Regression

    model = RidgeClassifier(alpha=10)
    model.fit(X_train, y_train)

    return model


def ridge_save_pickle(model):
    # Save pickle file
    output = open('pickle/ridge_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
