import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    # Load data
    with open('pickle/X_labeled.pkl', 'rb') as data_labeled:
        X = pickle.load(data_labeled)

    with open('pickle/y.pkl', 'rb') as labels:
        y = np.array(pickle.load(labels)).ravel()

    # Standardize X
    X_std = standardize(X)

    # Split test and train data
    (X_train, X_test, y_train, y_test) = train_test_split(X, y,
                                                          test_size=0.2,
                                                          random_state=1)
    (X_std_train, X_std_test,
     y_std_train, y_std_test) = train_test_split(X_std, y,
                                                 test_size=0.2,
                                                 random_state=1)

    # Load the feature sets
    top_feats = np.load('pickle/top_features.npz')['arr_0']
    rf_feat = top_feats[:200]
    ab_feat = top_feats[:300]
    gb_feat = top_feats[:300]
    knn_feat = top_feats[:13]
    nb_feat = top_feats[:5]
    gnb_feat = top_feats[:13]
    svm_feat = top_feats[:300]
    lr_feat = top_feats[:200]

    # Train the individual models
    rf_results = random_forest(X_train[rf_feat], y_train,
                               X_test[rf_feat], y_test)
    ab_results = adaboost(X_std_train[ab_feat], y_std_train,
                          X_std_test[ab_feat], y_std_test)
    gb_results = gradient_boosting(X_std_train[gb_feat], y_std_train,
                                   X_std_test[gb_feat], y_std_test)
    knn_results = knn(X_std_train[knn_feat], y_std_train,
                      X_std_test[knn_feat], y_std_test)
    nb_results = naive_bayes(X_train[nb_feat], y_train,
                             X_test[nb_feat], y_test)
    gnb_results = gaussian_naive_bayes(X_std_train[gnb_feat], y_std_train,
                                       X_std_test[gnb_feat], y_std_test)
    svm_results = svm(X_std_train[svm_feat], y_std_train,
                      X_std_test[svm_feat], y_std_test)
    lr_results = logistic_regression(X_std_train[lr_feat], y_std_train,
                                     X_std_test[lr_feat], y_std_test)

    print('Saving all models')
    save_pickle([rf_results, ab_results, gb_results, knn_results, nb_results,
                 gnb_results, svm_results, lr_results, y_test],
                'pickle/ensemble_results.pkl')


def save_pickle(objects, filename):
    # Save pickle file
    output = open(filename, 'wb')
    print('Pickle dump')
    for item in objects:
        pickle.dump(item, output, protocol=4)
    output.close()


def standardize(X):
    print('Standardizing')
    # Standardize features
    feat = ['favorite_count', 'retweet_count', 'compound', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'all_caps', 'hour']

    scaler = StandardScaler()
    X_std = X.copy()
    cols = X[feat].columns
    scaler.fit(X[feat])
    X_std[feat] = pd.DataFrame(scaler.transform(
                               X[feat]),
                               index=X.index,
                               columns=cols)
    save_pickle([scaler], 'pickle/ensemble_scaler.pkl')
    return X_std


def random_forest(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training Random Forest')
    rf = RandomForestClassifier(max_depth=20,
                                max_features='sqrt',
                                max_leaf_nodes=None,
                                min_samples_leaf=2,
                                min_samples_split=2,
                                n_estimators=1000,
                                n_jobs=-1).fit(X_train, y_train)
    predicted = rf.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([rf], 'pickle/ensemble_rf.pkl')

    return predicted


def adaboost(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training AdaBoost')
    ab = AdaBoostClassifier(learning_rate=1.25,
                            n_estimators=40).fit(X_train, y_train)
    predicted = ab.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([ab], 'pickle/ensemble_ab.pkl')

    return predicted


def gradient_boosting(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training Gradient Boosting')
    gb = GradientBoostingClassifier(n_estimators=200,
                                    learning_rate=.1,
                                    max_depth=6,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    subsample=1,
                                    max_features=None
                                    ).fit(X_train, y_train)
    predicted = gb.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([gb], 'pickle/ensemble_gb.pkl')

    return predicted


def knn(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training K Nearest Neighbors')

    # Perform Principle Component Analysis
    pca = PCA(n_components=12)
    pca.fit(X_train)

    save_pickle([pca], 'pickle/ensemble_knn_pca.pkl')
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    predicted = knn.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([knn], 'pickle/ensemble_knn.pkl')

    return predicted


def naive_bayes(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training Multnomial Naive Bayes')
    nb = MultinomialNB(alpha=10).fit(X_train, y_train)
    predicted = nb.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([nb], 'pickle/ensemble_nb.pkl')

    return predicted


def gaussian_naive_bayes(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training Gaussian Naive Bayes')

    # Perform Principle Component Analysis
    pca = PCA(n_components=10)
    pca.fit(X_train)

    save_pickle([pca], 'pickle/ensemble_gnb_pca.pkl')

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    gnb = GaussianNB().fit(X_train, y_train)
    predicted = gnb.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([gnb], 'pickle/ensemble_gnb.pkl')

    return predicted


def svm(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training SVM')
    svm = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=0.0001, max_iter=10).fit(X_train, y_train)
    predicted = svm.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([svm], 'pickle/ensemble_svm.pkl')

    return predicted


def logistic_regression(X_train, y_train, X_test, y_test):
    print('-------------------------------')
    print()
    print('Training Logistic Regression')
    lr = LogisticRegression(C=.05).fit(X_train, y_train)
    predicted = lr.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))
    print()

    save_pickle([lr], 'pickle/ensemble_lr.pkl')

    return predicted


if __name__ == '__main__':
    main()
