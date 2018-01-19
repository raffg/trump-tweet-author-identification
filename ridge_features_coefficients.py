import numpy as np
import pandas as pd
from src.load_pickle import load_pickle
from src.cross_val_data import cross_val_data
from src.standardize import standardize
from logistic_regression import lr


def main():
    coef = run_model_logistic_regression('pickle/data.pkl')
    top_features = np.load('pickle/top_features.npz')['arr_0']
    feats = [(x[0]) for x in list(top_features)]
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


def run_model_logistic_regression(file):
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle(file)

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08']

    (X_train, X_train_tfidf, X_train_pos, X_train_ner,
     X_test, X_test_tfidf, X_test_pos, X_test_ner) = cross_val_data(X_train,
                                                                    X_val,
                                                                    X_test)
    (X_train, X_test) = standardize(feat, X_train, X_test)

    # Concatenate all training DataFrames
    X_train = pd.concat([X_train, X_train_tfidf,
                         X_train_pos, X_train_ner], axis=1)
    X_test = pd.concat([X_test, X_test_tfidf,
                        X_test_pos, X_test_ner], axis=1)
    y_train = pd.concat([y_train, y_val], axis=0)

    model = lr(np.array(X_train),
               np.array(y_train).ravel())

    return model[3][0]


if __name__ == '__main__':
    main()
