import numpy as np
import pandas as pd
from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from src.standardize import standardize
from logistic_regression import lr


def main():
    coef = run_model_logistic_regression()
    feats = np.load('pickle/top_features.npz')['arr_0']
    features_coefs = list(zip(feats, coef))
    np.savez('pickle/features_coefs.npz', features_coefs)

    trump_feats = []
    not_trump_feats = []

    for item in features_coefs:
        if item[1] > 0:
            trump_feats.append(item[0])
        else:
            not_trump_feats.append(item[0])

    print(trump_feats)
    print()
    print(not_trump_feats)

    np.savez('pickle/trump_feats.npz',
             trump_feats=trump_feats,
             not_trump_feats=not_trump_feats)


def run_model_logistic_regression():
    X_train = pd.read_pickle('pickle/train_all_std.pkl')
    y_train = pd.read_pickle('pickle/y_train_all_std.pkl')

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

    # Remove non-numeric features
    X_train = X_train.drop(drop, axis=1)

    model = lr(np.array(X_train),
               np.array(y_train).ravel())

    return model[3][0]


if __name__ == '__main__':
    main()
