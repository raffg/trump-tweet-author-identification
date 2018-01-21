import pandas as pd
import numpy as np
import pickle
from src.standardize import standardize
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_ridge_regression()
    # ridge_save_pickle()


def run_model_ridge_regression():
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

    ridge_all_features = ridge(X_train[feat], X_val[feat], y_train, y_val)
    print()

    whole_train = X_train.drop(drop, axis=1)
    whole_val = X_val.drop(drop, axis=1)

    ridge_whole = ridge(whole_train, whole_val,
                        y_train, y_val)
    print()

    top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:100])
    train_feat = []
    val_feat = []
    for feat in top_feat:
        if feat in whole_train.columns:
            train_feat.append(feat)
        if feat in whole_val.columns:
            val_feat.append(feat)

    print('condensed model')
    condensed_train = whole_train[train_feat]
    condensed_val = whole_val[val_feat]

    ridge_condensed = ridge(condensed_train[train_feat],
                            condensed_val[val_feat],
                            y_train, y_val)
    print()


def ridge(X_train, X_val, y_train, y_val):
    # Ridge Logistic Regression

    model = RidgeClassifier(alpha=10)
    model.fit(X_train, y_train)
    predicted = model.predict(X_val)
    print()
    print('Accuracy: ', accuracy_score(y_val, predicted))
    print('Precision: ', precision_score(y_val, predicted))
    print('Recall: ', recall_score(y_val, predicted))

    # Save pickle file
    output = open('pickle/ridge_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

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
