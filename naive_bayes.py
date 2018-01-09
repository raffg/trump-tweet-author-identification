import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from sklearn.naive_bayes import MultinomialNB


def main():
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     y_train, y_val, y_test) = load_pickle()

    naive_bayes_accuracy = naive_bayes(np.array(X_train_tfidf),
                                       np.array(X_val_tfidf),
                                       np.array(y_train).ravel(),
                                       np.array(y_val).ravel())
    print('text accuracy: ', naive_bayes_accuracy)

    naive_bayes_pos_n_grams = naive_bayes(np.array(X_train_pos),
                                          np.array(X_val_pos),
                                          np.array(y_train).ravel(),
                                          np.array(y_val).ravel())
    print('pos accuracy: ', naive_bayes_pos_n_grams)

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'neg', 'neu', 'tweet_length', 'avg_sentence_length',
            'avg_word_length', 'quote', 'mentions', 'hashtags', 'urls',
            'is_quoted_retweet', 'all_caps']

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    feat_text_val = pd.concat([X_val[feat], X_val_tfidf], axis=1)

    naive_bayes_all_features_text = naive_bayes(np.array(feat_text_train),
                                                np.array(feat_text_val),
                                                np.array(y_train).ravel(),
                                                np.array(y_val).ravel())
    print('all features with text tf-idf accuracy: ',
          naive_bayes_all_features_text)

    feat_pos_train = pd.concat([X_train, X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val, X_val_pos], axis=1)
    naive_bayes_all_features_pos = naive_bayes(np.array(feat_pos_train),
                                               np.array(feat_pos_val),
                                               np.array(y_train).ravel(),
                                               np.array(y_val).ravel())
    print('all features with text & pos tf-idf accuracy: ',
          naive_bayes_all_features_pos)


def naive_bayes(X_train, X_val, y_train, y_val):
    # Basic Naive Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_val)
    accuracy = np.mean(predicted == y_val)
    return accuracy


if __name__ == '__main__':
    main()
