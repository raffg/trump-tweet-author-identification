import pandas as pd
import numpy as np
import pickle
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    flynn()
    example_tweets()
    # samples = run_samples()
    # accuracies(samples)


def predict_tweet(created_at):
    with open('pickle/random_forest_model.pkl', 'rb') as rf_f:
        rf = pickle.load(rf_f)

    with open('pickle/lr_model.pkl', 'rb') as lr_f:
        lr = pickle.load(lr_f)

    with open('pickle/naive_bayes_model.pkl', 'rb') as nb_f:
        nb = pickle.load(nb_f)

    with open('pickle/gaussian_nb_model.pkl', 'rb') as gnb_f:
        gnb = pickle.load(gnb_f)

    with open('pickle/adaboost_model.pkl', 'rb') as ab_f:
        ab = pickle.load(ab_f)

    with open('pickle/knn_model.pkl', 'rb') as knn_f:
        knn = pickle.load(knn_f)

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
    gnb_feat = np.load('pickle/top_features.npz')['arr_0'][:13]
    knn_feat = np.load('pickle/top_features.npz')['arr_0'][:13]
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
    tweet_knn = tweet_std
    tweet_gnb = tweet_std

    knn_train = pd.read_pickle('pickle/train_all_std.pkl')
    knn_train = knn_train.drop(drop, axis=1)
    pca = PCA(n_components=12)
    pca.fit(knn_train[knn_feat])
    tweet_knn = pca.transform(tweet_knn[knn_feat])

    pca = PCA(n_components=10)
    pca.fit(knn_train[gnb_feat])
    tweet_gnb = pca.transform(tweet_gnb[gnb_feat])

    tweet_rf = rf.predict(tweet[rf_feat])
    tweet_ab = ab.predict(tweet_std[ab_feat])
    tweet_knn = knn.predict(tweet_knn)
    tweet_nb = nb.predict(tweet_std[nb_feat])
    tweet_gnb = gnb.predict(tweet_gnb)
    tweet_svm = svm.predict(tweet_std[svm_feat])
    tweet_lr = lr.predict(tweet_std[lr_feat])

    proba_lr = lr.predict_proba(tweet_std[lr_feat])
    proba_rf = rf.predict_proba(tweet[rf_feat])
    # proba_knn = knn.predict_proba(tweet_knn.reshape(1, -1))
    proba_ab = ab.predict_proba(tweet_std[ab_feat])
    proba_nb = nb.predict_proba(tweet_std[nb_feat])

    maj_list = [tweet_rf[0], tweet_svm[0], tweet_lr[0],  tweet_gnb[0],
                tweet_nb[0], tweet_ab[0], tweet_knn[0]]

    maj_list = [tweet_rf[0], tweet_knn[0], tweet_lr[0], tweet_ab[0],
                tweet_svm[0]]

    majority = 1 if sum(maj_list) >= len(maj_list) / 2 else 0

    print('Random Forest prediction:         ', tweet_rf)
    print('AdaBoost prediction:              ', tweet_ab)
    print('KNN prediction:                   ', tweet_knn)
    print('Naive Bayes prediction:           ', tweet_nb)
    print('Gaussian Naive Bayes prediction:  ', tweet_gnb)
    print('SVM prediction:                   ', tweet_svm)
    print('Logistic Regression prediction:   ', tweet_lr)

    print('Random Forest Probabilities:      ', proba_rf)
    print('AdaBoost Probabilities:           ', proba_ab)
    # print('KNN Probabilities:                ', proba_knn)
    print('Naive Bayes Probabilities:        ', proba_nb)
    print('Logistic Regression probabilities:', proba_lr)
    print()
    print('Majority class:', majority)
    try:
        label = y.iat[X.index[X['created_at'] == created_at].tolist()[0], 0]
        print('True label:', label)
        return label, (tweet_rf, tweet_ab, tweet_knn, tweet_nb, tweet_gnb,
                       tweet_svm, tweet_lr)
    except Exception:
        pass
    print()

    return (tweet_rf, tweet_ab, tweet_knn, tweet_nb, tweet_gnb,
            tweet_svm, tweet_lr)


def run_samples():
    with open('pickle/X_labeled.pkl', 'rb') as data_labeled:
        X_labeled = pickle.load(data_labeled)

    with open('pickle/X_unlabeled.pkl', 'rb') as data_unlabeled:
        X_unlabeled = pickle.load(data_unlabeled)

    with open('pickle/y.pkl', 'rb') as labels:
        y = pickle.load(labels)

    X = pd.concat([X_labeled, X_unlabeled], axis=0).fillna(0)

    X = X[(X['created_at'] >= '2015-06-01') & (X['created_at'] < '2017-03-26')]

    sample = X.sample(n=10)

    results = []
    for index, row in sample.iterrows():
        result = predict_tweet(row['created_at'])
        results.append(result)
    return results


def accuracies(sample_results):
    models = [0, 1, 2, 3, 4, 5, 6]
    combos = (models + list(combinations(models, 3)) +
              list(combinations(models, 5)))
    y_true = [x[0] for x in sample_results]
    y_pred = defaultdict(list)

    for model in combos:
        model_pred = []
        for sample in sample_results:
            y_pred[model].append(1 if sum(maj_list) >=
                                 len(maj_list) / 2. else 0)


def flynn():
    print('Flynn tweet')
    flynn = predict_tweet('2017-12-02 17:14:13')


def example_tweets():
    print()
    print('WikiLeaks reveals Clinton camp’s work with ‘VERY friendly and '
          'malleable reporters’ #DrainTheSwamp #CrookedHillary '
          'https://t.co/bcYLslrxi0')
    tweet1 = predict_tweet('2016-10-21 22:46:37')
    print()

    print("Thanks for all of the accolades on my speech today - it's all about"
          ' the truth!"')
    tweet2 = predict_tweet('2013-03-15 23:33:34')
    print()

    print('Via @swan_investor by @Forbes: “The Trump Card: Make America Great'
          'Again” http://t.co/kWvbk5HtDr')
    tweet3 = predict_tweet('2015-05-13 17:50:05')
    print()

    print('Congratulations to Connecticut’s Erin Brady on being crowned the '
          '2013 @MissUSA! America will be well-represented in @MissUniverse!')
    tweet4 = predict_tweet('2013-06-17 18:13:52')
    print()

    print('We had a great News Conference at Trump Tower today. A couple of '
          "FAKE NEWS organizations were there but the people truly get what's"
          ' going on')
    tweet5 = predict_tweet('2017-01-12 04:01:38')
    print()

    print('According to a @gallupnews poll, over 60% think ObamaCare will make'
          ' things worse for taxpayers http://t.co/J375jNf1 ObamaCare is a '
          'T-A-X.')
    tweet6 = predict_tweet('2012-07-18 13:27:52')
    print()

    print('Thank you, Arizona! #Trump2016 #MakeAmericaGreatAgain #TrumpTrain')
    tweet7 = predict_tweet('2016-03-23 18:35:50')
    print()

    print("Lyin' Ted Cruz denied that he had anything to do with the G.Q. "
          "model photo post of Melania. That's why we call him Lyin' Ted!")
    tweet8 = predict_tweet('2016-03-23 14:22:51')
    print()

    print('I was relentless because, more often than you would think, sheer '
          'persistence is the difference between success and failure. NEVER '
          'GIVE UP!')
    tweet9 = predict_tweet('2014-10-08 12:09:04')
    print()

    print('"@MikeVega4: I have to say I fave no idea who @DannyZuker is but I'
          'know @realDonaldTrump is and he has great hotels #whoiszucker" '
          'TRUE.')
    tweet10 = predict_tweet('2013-06-13 01:48:07')
    print()

    print('Just tried watching Modern Family - written by a moron, really '
          'boring. Writer has the mind of a very dumb and backward child.'
          'Sorry Danny!')
    tweet11 = predict_tweet('2013-06-13 01:46:43')
    print()

    print('Whitey Bulger’s prosecution starts today.  Will be one of the most '
          'interesting and intriguing trials.')
    tweet12 = predict_tweet('2013-06-04 17:47:39')
    print()


if __name__ == '__main__':
    main()
