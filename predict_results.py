import pandas as pd
import numpy as np
import pickle
from itertools import combinations
from collections import defaultdict
from operator import itemgetter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
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

    with open('pickle/gradient_boosting_model.pkl', 'rb') as gb_f:
        gb = pickle.load(gb_f)

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
    gb_feat = np.load('pickle/top_features.npz')['arr_0'][:300]

    X = pd.concat([X_labeled, X_unlabeled], axis=0).fillna(0)
    X_std = pd.concat([X_labeled_std, X_unlabeled_std], axis=0).fillna(0)

    params = (rf, lr, nb, gnb, ab, gb, knn, svm, X, X_std, y, rf_feat, lr_feat,
              nb_feat, gnb_feat, knn_feat, svm_feat, ab_feat, gb_feat)

    # flynn(params)
    # example_tweets(params)
    sample_results = run_samples(params)
    save_data(sample_results, params)
    accuracies(sample_results[0], params)


def predict_tweet(params, created_at):
    (rf, lr, nb, gnb, ab, gb, knn, svm, X, X_std, y, rf_feat, lr_feat,
     nb_feat, gnb_feat, knn_feat, svm_feat, ab_feat, gb_feat) = params

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
    tweet_gb = gb.predict(tweet_std[gb_feat])

    proba_lr = lr.predict_proba(tweet_std[lr_feat])
    proba_rf = rf.predict_proba(tweet[rf_feat])
    proba_ab = ab.predict_proba(tweet_std[ab_feat])
    proba_gb = gb.predict_proba(tweet_std[gb_feat])
    proba_nb = nb.predict_proba(tweet_std[nb_feat])

    maj_list = [tweet_rf[0], tweet_svm[0], tweet_lr[0],  tweet_gnb[0],
                tweet_nb[0], tweet_ab[0], tweet_knn[0]]

    maj_list = [tweet_rf[0], tweet_knn[0], tweet_lr[0], tweet_ab[0],
                tweet_svm[0], tweet_gnb[0], tweet_gb[0]]

    majority = 1 if sum(maj_list) >= len(maj_list) / 2 else 0

    print('Random Forest prediction:         ', tweet_rf)
    print('AdaBoost prediction:              ', tweet_ab)
    print('Gradient Boosting prediction:     ', tweet_gb)
    print('KNN prediction:                   ', tweet_knn)
    print('Naive Bayes prediction:           ', tweet_nb)
    print('Gaussian Naive Bayes prediction:  ', tweet_gnb)
    print('SVM prediction:                   ', tweet_svm)
    print('Logistic Regression prediction:   ', tweet_lr)

    print('Random Forest Probabilities:      ', proba_rf)
    print('AdaBoost Probabilities:           ', proba_ab)
    print('Gradient Boosting Probabilities:  ', proba_gb)
    print('Naive Bayes Probabilities:        ', proba_nb)
    print('Logistic Regression probabilities:', proba_lr)
    print()
    print('Majority class:', majority)
    try:
        label = y.iat[X.index[X['created_at'] == created_at].tolist()[0], 0]
        print('True label:', label)
        return label, (tweet_rf[0], tweet_ab[0], tweet_gb[0], tweet_knn[0],
                       tweet_nb[0], tweet_gnb[0], tweet_svm[0], tweet_lr[0])
    except Exception:
        pass
    print()

    return (tweet_rf[0], tweet_ab[0], tweet_gb[0], tweet_knn[0],
            tweet_nb[0], tweet_gnb[0], tweet_svm[0], tweet_lr[0])


def run_samples(params):
    with open('pickle/X_labeled.pkl', 'rb') as data_labeled:
        X = pickle.load(data_labeled)

    with open('pickle/y.pkl', 'rb') as labels:
        y = pickle.load(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # sample = X.sample(n=10000)

    results = []
    n = 0
    N = len(X_train)
    for index, row in X_train.iterrows():
        n += 1
        print('Tweet #{} out of {}'.format(n, N))
        print(row['text'])
        print()
        result = predict_tweet(params, row['created_at'])
        print('-----------------------------------------------------------')
        results.append(result)

    results_test = []
    n = 0
    N = len(X_test)
    for index, row in X_test.iterrows():
        n += 1
        print('Tweet #{} out of {}'.format(n, N))
        print(row['text'])
        print()
        result = predict_tweet(params, row['created_at'])
        print('-----------------------------------------------------------')
        results_test.append(result)

    return results, results_test


def accuracies(sample_results):
    models = [0, 1, 2, 3, 4, 5, 6, 7]
    combos = (models + list(combinations(models, 3)) +
              list(combinations(models, 5)) + list(combinations(models, 7)))
    y_true = [x[0] for x in sample_results]

    y_pred = defaultdict(list)

    for model in combos:
        model_pred = []
        for sample in sample_results:
            ensemble = np.array(sample[1])[[model]]
            pred = 1 if sum(ensemble) > len(ensemble) / 2 else 0
            model_pred.append(pred)
        y_pred[(model)] = model_pred

    model_results = defaultdict(list)
    for key, value in y_pred.items():
        accuracy = accuracy_score(y_true, value)
        precision = precision_score(y_true, value)
        recall = recall_score(y_true, value)
        f1 = f1_score(y_true, value)
        model_results[key] = (accuracy, precision, recall, f1)

    sorted_results = [(v, k) for k, v in model_results.items()]
    sorted_results = sorted(model_results.items(), key=itemgetter(1),
                            reverse=True)
    for v, k in sorted_results[:10]:
        print(v, k)


def flynn(params):
    print()
    print('Flynn tweet')
    print()
    flynn = predict_tweet(params, '2017-12-02 17:14:13')
    print('-----------------------------------------------------------')


def example_tweets(params):
    print()
    print('WikiLeaks reveals Clinton camp’s work with ‘VERY friendly and '
          'malleable reporters’ #DrainTheSwamp #CrookedHillary '
          'https://t.co/bcYLslrxi0')
    print()
    tweet1 = predict_tweet(params, '2016-10-21 22:46:37')
    print('-----------------------------------------------------------')
    print()

    print("Thanks for all of the accolades on my speech today - it's all about"
          ' the truth!"')
    tweet2 = predict_tweet(params, '2013-03-15 23:33:34')
    print('-----------------------------------------------------------')
    print()

    print('Via @swan_investor by @Forbes: “The Trump Card: Make America Great '
          'Again” http://t.co/kWvbk5HtDr')
    tweet3 = predict_tweet(params, '2015-05-13 17:50:05')
    print('-----------------------------------------------------------')
    print()

    print('Congratulations to Connecticut’s Erin Brady on being crowned the '
          '2013 @MissUSA! America will be well-represented in @MissUniverse!')
    tweet4 = predict_tweet(params, '2013-06-17 18:13:52')
    print('-----------------------------------------------------------')
    print()

    print('We had a great News Conference at Trump Tower today. A couple of '
          "FAKE NEWS organizations were there but the people truly get what's"
          ' going on')
    print()
    tweet5 = predict_tweet(params, '2017-01-12 04:01:38')
    print('-----------------------------------------------------------')
    print()

    print('According to a @gallupnews poll, over 60% think ObamaCare will make'
          ' things worse for taxpayers http://t.co/J375jNf1 ObamaCare is a '
          'T-A-X.')
    print()
    tweet6 = predict_tweet(params, '2012-07-18 13:27:52')
    print('-----------------------------------------------------------')
    print()

    print('Thank you, Arizona! #Trump2016 #MakeAmericaGreatAgain #TrumpTrain')
    print()
    tweet7 = predict_tweet(params, '2016-03-23 18:35:50')
    print('-----------------------------------------------------------')
    print()

    print("Lyin' Ted Cruz denied that he had anything to do with the G.Q. "
          "model photo post of Melania. That's why we call him Lyin' Ted!")
    print()
    tweet8 = predict_tweet(params, '2016-03-23 14:22:51')
    print('-----------------------------------------------------------')
    print()

    print('I was relentless because, more often than you would think, sheer '
          'persistence is the difference between success and failure. NEVER '
          'GIVE UP!')
    print()
    tweet9 = predict_tweet(params, '2014-10-08 12:09:04')
    print('-----------------------------------------------------------')
    print()

    print('"@MikeVega4: I have to say I fave no idea who @DannyZuker is but I '
          'know @realDonaldTrump is and he has great hotels #whoiszucker" '
          'TRUE.')
    print()
    tweet10 = predict_tweet(params, '2013-06-13 01:48:07')
    print('-----------------------------------------------------------')
    print()

    print('Just tried watching Modern Family - written by a moron, really '
          'boring. Writer has the mind of a very dumb and backward child.'
          'Sorry Danny!')
    print()
    tweet11 = predict_tweet(params, '2013-06-13 01:46:43')
    print('-----------------------------------------------------------')
    print()

    print('Whitey Bulger’s prosecution starts today.  Will be one of the most '
          'interesting and intriguing trials.')
    print()
    tweet12 = predict_tweet(params, '2013-06-04 17:47:39')
    print('-----------------------------------------------------------')
    print()


def save_data(sample_results):
    X_train = [x[1] for x in sample_results[0]]
    y_train = [x[0] for x in sample_results[0]]
    X_test = [x[1] for x in sample_results[1]]
    y_test = [x[0] for x in sample_results[1]]
    np.savez('pickle/ensemble_predictions_X_train.npz', X_train)
    np.savez('pickle/ensemble_predictions_y_train.npz', y_train)
    np.savez('pickle/ensemble_predictions_X_test.npz', X_test)
    np.savez('pickle/ensemble_predictions_y_test.npz', y_test)


if __name__ == '__main__':
    main()
