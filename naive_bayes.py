import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_naive_bayes()


def run_model_naive_bayes():
    X_train = pd.read_pickle('pickle/train_all.pkl')
    X_val = pd.read_pickle('pickle/test_all.pkl')
    y_train = pd.read_pickle('pickle/y_train_all.pkl')
    y_val = pd.read_pickle('pickle/y_test_all.pkl')

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

    print('all features')
    naive_bayes_all_features = naive_bayes(np.array(X_train[feat]),
                                           np.array(X_val[feat]),
                                           np.array(y_train).ravel(),
                                           np.array(y_val).ravel())

    whole_train = X_train.drop(drop, axis=1)
    whole_val = X_val.drop(drop, axis=1)

    print('whole model')
    naive_bayes_whole = naive_bayes(np.array(whole_train),
                                    np.array(whole_val),
                                    np.array(y_train).ravel(),
                                    np.array(y_val).ravel())
    # naive_bayes_save_pickle(naive_bayes_whole)

    top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:5])
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
    naive_bayes_condensed = naive_bayes(np.array(condensed_train),
                                        np.array(condensed_val),
                                        np.array(y_train).ravel(),
                                        np.array(y_val).ravel())
    # naive_bayes_save_pickle(naive_bayes_condensed)


def naive_bayes(X_train, X_val, y_train, y_val):
    # Basic Naive Bayes
    nb = MultinomialNB(alpha=10).fit(X_train, y_train)
    predicted = nb.predict(X_val)
    accuracy_train = np.mean(nb.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)

    print('Accuracy: ', accuracy_score(y_val, predicted))
    print('Precision: ', precision_score(y_val, predicted))
    print('Recall: ', recall_score(y_val, predicted))
    print()

    return nb


def naive_bayes_save_pickle(model):
    # Save pickle file
    output = open('pickle/naive_bayes_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
