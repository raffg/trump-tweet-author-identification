import pandas as pd
import numpy as np
import pickle
from src.ridge_grid_scan import ridge_grid_scan
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def Ensemble(object):
    ''' This class represents the ensemble of models for Trump tweet authorship
    prediction

    Methods
    -------
    fit : fit the model to X and y data
    predict : predict the authoriship of an unlabled tweet_length

    Attributes
    ----------
    top_feats : Array of the features sorted by influence

    Returns
    -------
    self:
        The initialized GradientDescent object.
    '''

    def __init__(self):
        ''' Initialize the ensemble object
        '''
        self.model = None
        self.scaler = None
        self.knn_pca = None
        self.gnb_pca = None
        self.top_feats = None

    def fit(self, X_train, y_train):
        ''' Train the ensemble with X and y data

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The training data for the optimization.
        y: ndarray, shape (n_samples, ).
            The training response for the optimization.

        Returns
        -------
        self:
            The fit Ensemble object.
        '''
        # Columns to standardize
        feat = ['favorite_count', 'retweet_count', 'compound', 'anger',
                'anticipation', 'disgust', 'fear', 'joy', 'negative',
                'positive', 'sadness', 'surprise', 'trust', 'tweet_length',
                'avg_sentence_length', 'avg_word_length', 'commas',
                'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
                'ellipses', 'mentions', 'hashtags', 'urls', 'all_caps', 'hour']

        # Train the standard scaler
        standard_scaler(X_train, feat)

        # Train the PCA objects
        gnb_pca_calc
        knn_pca_calc

        X_train, X_std_train = prepare_data_for_fit(X_train)

        # Load the feature sets
        feature_list = ridge_grid_scan(X_train,
                                       np.array(y_train).ravel(),
                                       n=len(X_train.columns))
        self.top_feats = [(x[0]) for x in list(feature_list)]

        # Train the individual models
        data = first_stage_predict(X_train, y_train)

        X_train_dt = pd.DataFrame(data)

        X_train_dt['majority'] = X_train_dt.apply(majority, axis=1)

        self.model = decision_tree(X_train_dt, y_train)

        return self

    def predict(self, X):
        pass

    def standard_scaler(X, feat):
        # Standardize features
        self.scaler = StandardScaler()
        cols = X[feat].columns
        self.scaler.fit(X[feat])

    def standardize(X, feat):
        X_std = X
        X_std[feat] = pd.DataFrame(self.scaler.transform(
                                   X[feat]),
                                   index=X.index,
                                   columns=cols)
        return X_std

    def prepare_data_for_fit(self, X):
        ''' Processes the X data with all features and standardizes.
        '''
        X_std = standardize(X)
        pass

    def first_stage_predict(X, y):
        '''Calculate predictions for first stage of 9 models
        '''
        rf_feat = self.top_feats[:200]
        ab_feat = self.top_feats[:300]
        gb_feat = self.top_feats[:300]
        knn_feat = self.top_feats[:13]
        nb_feat = self.top_feats[:5]
        gnb_feat = self.top_feats[:13]
        svc_feat = self.top_feats[:50]
        svm_feat = self.top_feats[:300]
        lr_feat = self.top_feats[:200]

        rf_results = random_forest(X_train[rf_feat], y_train)
        ab_results = adaboost(X_std_train[ab_feat], y_train)
        gb_results = gradient_boosting(X_std_train[gb_feat], y_train)
        knn_results = knn(X_std_train[knn_feat], y_train)
        nb_results = naive_bayes(X_train[nb_feat], y_train)
        gnb_results = gaussian_naive_bayes(X_std_train[gnb_feat], y_train)
        svc_results = svc(X_std_train[svc_feat], y_train)
        svm_results = svm(X_std_train[svm_feat], y_train)
        lr_results = logistic_regression(X_std_train[lr_feat], y_train)

        return {'rf': rf_results, 'ab': ab_results, 'gb': gb_results,
                'knn': knn_results, 'nb': nb_results, 'gnb': gnb_results,
                'svc': svc_results, 'svm': svm_results, 'lr': lr_results}

    def random_forest(X_train, y_train):
        rf = RandomForestClassifier(max_depth=20,
                                    max_features='sqrt',
                                    max_leaf_nodes=None,
                                    min_samples_leaf=2,
                                    min_samples_split=2,
                                    n_estimators=1000,
                                    n_jobs=-1).fit(X_train, y_train)
        predicted = rf.predict(X_train)

        return predicted

    def adaboost(X_train, y_train):
        ab = AdaBoostClassifier(learning_rate=1.25,
                                n_estimators=40).fit(X_train, y_train)
        predicted = ab.predict(X_train)

        return predicted

    def gradient_boosting(X_train, y_train):
        gb = GradientBoostingClassifier(n_estimators=200,
                                        learning_rate=.1,
                                        max_depth=6,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        subsample=1,
                                        max_features=None
                                        ).fit(X_train, y_train)
        predicted = gb.predict(X_train)

        return predicted

    def knn(X_train, y_train):
        X_train = self.knn_pca.transform(X_train)

        knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
        predicted = knn.predict(X_train)

        return predicted

    def knn_pca_calc(X_train):
        # Perform Principle Component Analysis
        pca = PCA(n_components=12)
        pca.fit(X_train)
        self.knn_pca = pca

    def naive_bayes(X_train, y_train):
        nb = MultinomialNB(alpha=10).fit(X_train, y_train)
        predicted = nb.predict(X_train)

        return predicted

    def gaussian_naive_bayes(X_train, y_train):
        X_train = self.gnb_pca.transform(X_train)

        gnb = GaussianNB().fit(X_train, y_train)
        predicted = gnb.predict(X_train)

        return predicted, pca

    def gnb_pca_calc(X_train):
        # Perform Principle Component Analysis
        pca = PCA(n_components=10)
        pca.fit(X_train)
        self.gnb_pca = pca

    def svc(X_train, y_train):
        svc = SVC(C=100,
                  coef0=1,
                  degree=2,
                  gamma='auto',
                  kernel='poly',
                  shrinking=False).fit(X_train, y_train)
        predicted = svc.predict(X_train)

        return predicted

    def svm(X_train, y_train):
        svm = SGDClassifier(loss='hinge', penalty='l2',
                            alpha=0.0001, max_iter=10).fit(X_train, y_train)
        predicted = svm.predict(X_train)

        return predicted

    def logistic_regression(X_train, y_train):
        lr = LogisticRegression(C=.05).fit(X_train, y_train)
        predicted = lr.predict(X_train)

        return predicted

    def decision_tree(X_train, y_train):
        dt = DecisionTreeClassifier(criterion='gini',
                                    max_depth=None,
                                    min_weight_fraction_leaf=0.001,
                                    splitter='best')
        dt.fit(X_train, y_train)

        return dt

    def majority(row):
        val = 1 if (row['rf'] + row['ab'] + row['gb'] + row['knn'] + row['nb']
                    + row['gnb'] + row['svc'] + row['svm'] + row['lr']
                    ) > 3 else 0
        return val
