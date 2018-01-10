import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from sklearn.linear_model import SGDClassifier


def main():
    run_model_svm(data.pkl)


def run_model_svm(file):
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     y_train, y_val, y_test) = load_pickle(file)

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'negative', 'neutral', 'positive', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'quote', 'mentions',
            'hashtags', 'urls', 'is_quoted_retweet', 'all_caps']

    svm_all_features = svm(np.array(X_train[feat]),
                           np.array(X_val[feat]),
                           np.array(y_train).ravel(),
                           np.array(y_val).ravel())
    print('all features accuracy: ', svm_all_features)

    svm_text_accuracy = svm(np.array(X_train_tfidf),
                            np.array(X_val_tfidf),
                            np.array(y_train).ravel(),
                            np.array(y_val).ravel())
    print('text accuracy: ', svm_text_accuracy)

    svm_pos_n_grams = svm(np.array(X_train_pos),
                          np.array(X_val_pos),
                          np.array(y_train).ravel(),
                          np.array(y_val).ravel())
    print('pos accuracy: ', svm_pos_n_grams)

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    feat_text_val = pd.concat([X_val[feat], X_val_tfidf], axis=1)

    svm_all_features_text = svm(np.array(feat_text_train),
                                np.array(feat_text_val),
                                np.array(y_train).ravel(),
                                np.array(y_val).ravel())
    print('all features with text tf-idf accuracy: ',
          svm_all_features_text)

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val[feat], X_val_pos], axis=1)
    svm_all_features_pos = svm(np.array(feat_pos_train),
                               np.array(feat_pos_val),
                               np.array(y_train).ravel(),
                               np.array(y_val).ravel())
    print('all features with pos tf-idf accuracy: ',
          svm_all_features_pos)

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val[feat], X_val_pos], axis=1)
    svm_all_features_pos = svm(np.array(feat_pos_train),
                               np.array(feat_pos_val),
                               np.array(y_train).ravel(),
                               np.array(y_val).ravel())
    print('all features with pos tf-idf accuracy: ',
          svm_all_features_pos)

    vader = svm(np.array(X_train[['compound', 'negative',
                                  'neutral', 'positive']]),
                np.array(X_val[['compound', 'negative',
                                'neutral', 'positive']]),
                np.array(y_train).ravel(),
                np.array(y_val).ravel())
    print('vader features accuracy: ', vader)


def svm(X_train, X_val, y_train, y_val):
    # Basic Naive Bayes
    clf = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e3, max_iter=50).fit(X_train, y_train)
    predicted = clf.predict(X_val)
    accuracy_train = np.mean(clf.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)
    return accuracy_train, accuracy_test


if __name__ == '__main__':
    main()
