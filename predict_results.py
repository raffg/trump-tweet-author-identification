import pandas as pd
import numpy as np
import pickle


def main():
    print('Flynn tweet')
    flynn = predict_tweet('2017-12-02 17:14:13')
    print()
    tweet1 = predict_tweet('2016-10-21 22:46:37')
    print()
    tweet2 = predict_tweet('2013-03-15 23:33:34')
    print()
    tweet3 = predict_tweet('2015-05-13 17:50:05')
    print()
    tweet4 = predict_tweet('2013-06-17 18:13:52')
    print()
    tweet5 = predict_tweet('2017-01-12 04:01:38')


def predict_tweet(created_at):
    with open('pickle/random_forest_model.pkl', 'rb') as rf_f:
        rf = pickle.load(rf_f)

    with open('pickle/lr_model.pkl', 'rb') as lr_f:
        lr = pickle.load(lr_f)

    with open('pickle/adaboost_model.pkl', 'rb') as ab_f:
        ab = pickle.load(ab_f)

    with open('pickle/naive_bayes_model.pkl', 'rb') as nb_f:
        nb = pickle.load(nb_f)

    with open('pickle/svm_model.pkl', 'rb') as svm_f:
        svm = pickle.load(svm_f)

    with open('pickle/X_labeled.pkl', 'rb') as data_labeled:
        X_labeled = pickle.load(data_labeled)

    with open('pickle/X_unlabeled.pkl', 'rb') as data_unlabeled:
        X_unlabeled = pickle.load(data_unlabeled)

    with open('pickle/X_labeled_std.pkl', 'rb') as data_labeled_std:
        X_labeled_std = pickle.load(data_labeled_std)

    with open('pickle/X_unlabeled_std.pkl', 'rb') as data_unlabeled_std:
        X_unlabeled_std = pickle.load(data_unlabeled_std)

    with open('pickle/y.pkl', 'rb') as labels:
        y = pickle.load(labels)

    rf_feat = np.load('pickle/top_features.npz')['arr_0'][:200]
    lr_feat = np.load('pickle/top_features.npz')['arr_0'][:200]
    nb_feat = np.load('pickle/top_features.npz')['arr_0'][:5]
    svm_feat = np.load('pickle/top_features.npz')['arr_0'][:300]
    ab_feat = np.load('pickle/top_features.npz')['arr_0'][:300]

    X = pd.concat([X_labeled, X_unlabeled], axis=0).fillna(0)
    X_std = pd.concat([X_labeled_std, X_unlabeled_std], axis=0).fillna(0)

    drop = ['created_at', 'id_str', 'in_reply_to_user_id_str', 'tweetokenize',
            'text', 'pos', 'ner']

    tweet = X[X['created_at'] == created_at]
    tweet_std = X_std[X_std['created_at'] == created_at]

    tweet = tweet.drop(drop, axis=1)
    tweet_std = tweet_std.drop(drop, axis=1)

    tweet_rf = rf.predict(tweet[rf_feat])
    tweet_ab = ab.predict(tweet_std[ab_feat])
    tweet_nb = nb.predict(tweet[nb_feat])
    tweet_svm = svm.predict(tweet_std[svm_feat])
    tweet_lr = lr.predict(tweet_std[lr_feat])
    proba_lr = lr.predict_proba(tweet_std[lr_feat])

    majority = 1 if sum([tweet_rf[0], tweet_svm[0], tweet_lr[0],
                         tweet_ab[0], tweet_nb[0]]) >= 3 else 0

    print('Random Forest prediction:', tweet_rf)
    print('AdaBoost prediction:', tweet_ab)
    print('Naive Bayes prediction:', tweet_nb)
    print('SVM prediction:', tweet_svm)
    print('Logistic Regression prediction:', tweet_lr)
    print('Logistic Regression probabilities:', proba_lr)
    print()
    print('Majority class:', majority)
    try:
        label = y.iat[X.index[X['created_at'] == created_at].tolist()[0], 0]
        print('True label:', label)
    except Exception:
        pass

    return tweet_rf, tweet_ab, tweet_nb, tweet_svm, tweet_lr, majority


if __name__ == '__main__':
    main()
