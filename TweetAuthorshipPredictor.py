import pandas as pd
import numpy as np
import pickle
from src.ridge_grid_scan import ridge_grid_scan
from src.feature_pipeline import feature_pipeline
from src.load_data import load_json_list, apply_date_mask
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    data_list = (['data/condensed_2009.json',
                  'data/condensed_2010.json',
                  'data/condensed_2011.json',
                  'data/condensed_2012.json',
                  'data/condensed_2013.json',
                  'data/condensed_2014.json',
                  'data/condensed_2015.json',
                  'data/condensed_2016.json',
                  'data/condensed_2017.json'])

    print('Loading data...')
    df = load_json_list(data_list)
    df = df.sort_values('created_at').reset_index(drop=True)
    df2 = apply_date_mask(df, 'created_at', '2009-01-01', '2017-03-26')

    y = pd.DataFrame(np.where(df2['source'] == 'Twitter for Android', 1, 0))
    X = df2.copy()
    save_pickle(y, 'ensemble/y_train.pkl')

    trump = TweetAuthorshipPredictor()
    trump.fit(X, y)

    save_pickle(trump, 'ensemble/trump.pkl')


class TweetAuthorshipPredictor(object):
    ''' This class represents the ensemble of models for tweet authorship
    prediction

    Parameters
    ----------
    featurized: boolean, option (default=False)
        Boolean indicating if the X data has already been featurized.
        Use True if sending featurized data to the class.

    Methods
    -------
    fit : fit the model to X and y data
    predict : predict the authoriship of an unlabled tweet_length
    get_top_features : returns a list of the features ordered by influence

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
        # Save the individual ensemble models
        self.rf = None
        self.ab = None
        self.gb = None
        self.knn = None
        self.nb = None
        self.gnb = None
        self.svc = None
        self.svm = None
        self.lr = None
        self.dt = None

        # Save the data processing objects
        self.top_feats = None
        self.scaler = None
        self.knn_pca = None
        self.gnb_pca = None
        self.tfidf_text = None
        self.tfidf_ner = None
        self.tfidf_pos = None

        # Columns to standardize
        self.std = ['compound', 'anger', 'anticipation', 'disgust', 'fear',
                    'joy', 'negative', 'positive', 'sadness', 'surprise',
                    'trust', 'avg_sentence_length', 'avg_word_length',
                    'commas', 'semicolons', 'exclamations', 'periods',
                    'questions', 'quotes', 'ellipses', 'mentions',
                    'hashtags', 'urls', 'all_caps', 'hour', 'random_caps']

        # Columns to train on prior to tf-idf
        self.feat = ['created_at', 'is_retweet', 'text', 'is_reply',
                     'compound', 'v_negative', 'v_neutral', 'v_positive',
                     'anger', 'anticipation', 'disgust', 'fear', 'joy',
                     'negative', 'positive', 'sadness', 'surprise', 'trust',
                     'avg_sentence_length', 'avg_word_length', 'commas',
                     'semicolons', 'exclamations', 'periods', 'questions',
                     'quotes', 'ellipses', 'mentions', 'hashtags', 'urls',
                     'is_quoted_retweet', 'all_caps', 'tweetstorm', 'hour',
                     'hour_20_02', 'hour_14_20', 'hour_08_14', 'hour_02_08',
                     'day_of_week', 'weekend', 'random_caps', 'start_mention',
                     'ner', 'pos']

        # tf-idf column names
        self.text_cols = None
        self.pos_cols = None
        self.ner_cols = None

    def fit(self, X_train, y_train):
        ''' Train the ensemble with X and y data

        Parameters
        ----------
        X: Pandas DataFrame, shape (n_samples, n_features)
            The training data.
        y: Pandas DataFrame, shape (n_samples, ).
            The training response for the optimization.

        Returns
        -------
        self:
            The fit Ensemble object.
        '''
        # Featurize the X data
        X_train, X_std_train = self._prepare_data_for_fit(X_train)

        save_pickle(X_train, 'ensemble/X_train.pkl')
        save_pickle(X_std_train, 'ensemble/X_std_train.pkl')
        # X_train = load_pickle('twitterbot_pickles/X_train.pkl')
        # X_std_train = load_pickle('twitterbot_pickles/X_std_train.pkl')

        drop = ['created_at', 'text', 'pos', 'ner']

        # self.tfidf_pos = load_pickle('twitterbot_pickles/tfidf_pos.pkl')
        # self.tfidf_ner = load_pickle('twitterbot_pickles/tfidf_ner.pkl')
        # self.tfidf_text = load_pickle('twitterbot_pickles/tfidf_text.pkl')
        # self.scaler = load_pickle('twitterbot_pickles/scaler.pkl')
        self.text_cols = self.tfidf_text.get_feature_names()
        self.ner_cols = self.tfidf_ner.get_feature_names()
        self.pos_cols = self.tfidf_pos.get_feature_names()
        #
        # # Remove non-numeric features
        X_train = X_train.drop(drop, axis=1)
        X_std_train = X_std_train.drop(drop, axis=1)

        # Load the feature sets
        feature_list = ridge_grid_scan(X_train,
                                       np.array(y_train).ravel(),
                                       n=len(X_train.columns))
        self.top_feats = [(x[0]) for x in list(feature_list)]
        save_pickle(self.top_feats, 'ensemble/top_feats.pkl')
        # self.top_feats = load_pickle('twitterbot_pickles/top_feats.pkl')

        # Train the PCA objects
        self._gnb_pca_calc(X_std_train[self.top_feats[:13]])
        self._knn_pca_calc(X_std_train[self.top_feats[:13]])
        save_pickle(self.gnb_pca, 'ensemble/gnb_pca.pkl')
        save_pickle(self.knn_pca, 'ensemble/knn_pca.pkl')
        # self.gnb_pca = load_pickle('twitterbot_pickles/gnb_pca.pkl')
        # self.knn_pca = load_pickle('twitterbot_pickles/knn_pca.pkl')

        # Train the individual models
        data = self._first_stage_train(X_train, X_std_train,
                                       np.array(y_train).ravel())
        X_train_dt = pd.DataFrame(data)

        X_train_dt['majority'] = X_train_dt.apply(self._majority, axis=1)
        save_pickle(X_train_dt, 'ensemble/X_train_dt.pkl')
        # X_train_dt = load_pickle('twitterbot_pickles/X_train_dt.pkl')

        self.dt = self._decision_tree(X_train_dt, np.array(y_train).ravel())

        return self

    def predict(self, X):
        '''Return a label for prediction of the authoriship of the tweet X

        Parameters
        ----------
        X: 2d Pandas DataFrame
            The feature matrix

        Returns
        -------
        y: 1 or 0
            Predicted label
        '''
        X, X_std = self._prepare_data_for_predict(X)
        data, probabilities = self._first_stage_predict(X, X_std)
        X_dt = pd.DataFrame(data)
        X_dt['majority'] = X_dt.apply(self._majority, axis=1)

        prediction = self.dt.predict(X_dt)
        proba_list = []
        for key, value in probabilities.items():
            if data[key] == prediction:
                proba_list.append(value[0][prediction])
        proba = np.mean(proba_list)

        return self.dt.predict(X_dt), proba

    def get_top_features(self):
        '''Returns a list of the features ordered by influence
        '''
        return self.top_feats

    def _standard_scaler(self, X):
        # Standardize features
        print('Calculating standardization')
        self.scaler = StandardScaler()
        cols = X.columns
        self.scaler.fit(X)
        save_pickle(self.scaler, 'ensemble/scaler.pkl')
        # self.scaler = load_pickle('twitterbot_pickles/scaler.pkl')

    def _standardize(self, X):
        X_std = X.copy()
        cols = X[self.std].columns
        X_std[self.std] = pd.DataFrame(self.scaler.transform(
                                       X[self.std]),
                                       index=X.index,
                                       columns=cols)
        return X_std

    def _prepare_data_for_fit(self, X):
        ''' Processes the X data with all features, saves tf-idf vectorizers,
        and standardizes.
        '''
        # Create new feature columns
        X = feature_pipeline(X, verbose=True)
        X = self._tfidf_fit_transform(X[self.feat])
        self._standard_scaler(X[self.std])
        X_std = self._standardize(X)

        return X, X_std

    def _prepare_data_for_predict(self, X):
        ''' Processes the X data with all features and standardizes.
        '''
        # Create new feature columns
        X = feature_pipeline(X)
        X = self._tfidf_transform(X[self.feat])
        X_std = self._standardize(X)

        return X, X_std

    def _first_stage_train(self, X_train, X_std_train, y_train):
        '''Train models in first stage of 9 models
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

        rf_results = self._random_forest(X_train[rf_feat], y_train)
        ab_results = self._adaboost(X_std_train[ab_feat], y_train)
        gb_results = self._gradient_boosting(X_std_train[gb_feat], y_train)
        knn_results = self._knn(X_std_train[knn_feat], y_train)
        nb_results = self._naive_bayes(X_train[nb_feat], y_train)
        gnb_results = self._gaussian_naive_bayes(X_std_train[gnb_feat],
                                                 y_train)
        svc_results = self._svc(X_std_train[svc_feat], y_train)
        svm_results = self._svm(X_std_train[svm_feat], y_train)
        lr_results = self._logistic_regression(X_std_train[lr_feat], y_train)

        return {'rf': rf_results, 'ab': ab_results, 'gb': gb_results,
                'knn': knn_results, 'nb': nb_results, 'gnb': gnb_results,
                'svc': svc_results, 'svm': svm_results, 'lr': lr_results}

    def _first_stage_predict(self, X, X_std):
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

        X_knn = self.knn_pca.transform(X_std[knn_feat])
        X_gnb = self.gnb_pca.transform(X_std[gnb_feat])

        rf_results = self.rf.predict(X[rf_feat])
        ab_results = self.ab.predict(X_std[ab_feat])
        gb_results = self.gb.predict(X_std[gb_feat])
        knn_results = self.knn.predict(X_knn)
        nb_results = self.nb.predict(X[nb_feat])
        gnb_results = self.gnb.predict(X_gnb)
        svc_results = self.svc.predict(X_std[svc_feat])
        svm_results = self.svm.predict(X_std[svm_feat])
        lr_results = self.lr.predict(X_std[lr_feat])

        data = {'rf': rf_results, 'ab': ab_results, 'gb': gb_results,
                'knn': knn_results, 'nb': nb_results, 'gnb': gnb_results,
                'svc': svc_results, 'svm': svm_results, 'lr': lr_results}

        rf_predict = self.rf.predict_proba(X[rf_feat])
        ab_predict = self.ab.predict_proba(X_std[ab_feat])
        gb_predict = self.gb.predict_proba(X_std[gb_feat])
        knn_predict = self.knn.predict_proba(X_knn)
        nb_predict = self.nb.predict_proba(X[nb_feat])
        gnb_predict = self.gnb.predict_proba(X_gnb)
        lr_predict = self.lr.predict_proba(X_std[lr_feat])

        probabilities = {'rf': rf_predict, 'ab': ab_predict, 'gb': gb_predict,
                         'knn': knn_predict, 'nb': nb_predict,
                         'gnb': gnb_predict, 'lr': lr_predict}

        for key, value in probabilities.items():
            print(key, value)

        for key, value in data.items():
            print(key, value)

        return data, probabilities

    def _random_forest(self, X_train, y_train):
        print('Running Random Forest')
        rf = RandomForestClassifier(max_depth=20,
                                    max_features='sqrt',
                                    max_leaf_nodes=None,
                                    min_samples_leaf=2,
                                    min_samples_split=2,
                                    n_estimators=1000,
                                    n_jobs=-1).fit(X_train, y_train)
        predicted = rf.predict(X_train)
        self.rf = rf
        save_pickle(self.rf, 'ensemble/rf.pkl')
        # self.rf = load_pickle('twitterbot_pickles/rf.pkl')
        return predicted

    def _adaboost(self, X_train, y_train):
        print('Running AdaBoost')
        ab = AdaBoostClassifier(learning_rate=1.25,
                                n_estimators=40).fit(X_train, y_train)
        predicted = ab.predict(X_train)
        self.ab = ab
        save_pickle(self.ab, 'ensemble/ab.pkl')
        # self.ab = load_pickle('twitterbot_pickles/ab.pkl')
        return predicted

    def _gradient_boosting(self, X_train, y_train):
        print('Running Gradient Boosting')
        gb = GradientBoostingClassifier(n_estimators=200,
                                        learning_rate=.1,
                                        max_depth=6,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        subsample=1,
                                        max_features=None
                                        ).fit(X_train, y_train)
        predicted = gb.predict(X_train)
        self.gb = gb
        save_pickle(self.gb, 'ensemble/gb.pkl')
        # self.gb = load_pickle('twitterbot_pickles/gb.pkl')
        return predicted

    def _knn(self, X_train, y_train):
        print('Running K Nearest Neighbors')
        X_train = self.knn_pca.transform(X_train)
        knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
        predicted = knn.predict(X_train)
        self.knn = knn
        save_pickle(self.knn, 'ensemble/knn.pkl')
        # self.knn = load_pickle('twitterbot_pickles/knn.pkl')
        return predicted

    def _knn_pca_calc(self, X_train):
        # Perform Principle Component Analysis
        print('Performing PCA on K Nearest Neighbors')
        pca = PCA(n_components=12)
        pca.fit(X_train)
        self.knn_pca = pca

    def _naive_bayes(self, X_train, y_train):
        print('Running Multinomial Naive Bayes')
        nb = MultinomialNB(alpha=10).fit(X_train, y_train)
        predicted = nb.predict(X_train)
        self.nb = nb
        save_pickle(self.nb, 'ensemble/nb.pkl')
        # self.nb = load_pickle('twitterbot_pickles/nb.pkl')
        return predicted

    def _gaussian_naive_bayes(self, X_train, y_train):
        print('Running Gaussian Naive Bayes')
        X_train = self.gnb_pca.transform(X_train)
        gnb = GaussianNB().fit(X_train, y_train)
        predicted = gnb.predict(X_train)
        self.gnb = gnb
        save_pickle(self.gnb, 'ensemble/gnb.pkl')
        # self.gnb= load_pickle('twitterbot_pickles/gnb.pkl')
        return predicted

    def _gnb_pca_calc(self, X_train):
        # Perform Principle Component Analysis
        print('Performing PCA on Gaussian Naive Bayes')
        pca = PCA(n_components=10)
        pca.fit(X_train)
        self.gnb_pca = pca

    def _svc(self, X_train, y_train):
        print('Running Support Vector Classifier')
        svc = SVC(C=100,
                  coef0=1,
                  degree=2,
                  gamma='auto',
                  kernel='poly',
                  shrinking=False).fit(X_train, y_train)
        predicted = svc.predict(X_train)
        self.svc = svc
        save_pickle(self.svc, 'ensemble/svc.pkl')
        # self.svc = load_pickle('twitterbot_pickles/svc.pkl')
        return predicted

    def _svm(self, X_train, y_train):
        print('Running Support Vector Machine')
        svm = SGDClassifier(loss='hinge', penalty='l2',
                            alpha=0.0001, max_iter=10).fit(X_train, y_train)
        predicted = svm.predict(X_train)
        self.svm = svm
        save_pickle(self.svm, 'ensemble/svm.pkl')
        # self.svm = load_pickle('twitterbot_pickles/svm.pkl')
        return predicted

    def _logistic_regression(self, X_train, y_train):
        print('Running Logistic Regression')
        lr = LogisticRegression(C=.05).fit(X_train, y_train)
        predicted = lr.predict(X_train)
        self.lr = lr
        save_pickle(self.lr, 'ensemble/lr.pkl')
        # self.lr = load_pickle('twitterbot_pickles/lr.pkl')
        return predicted

    def _decision_tree(self, X_train, y_train):
        print('Running Decision Tree')
        dt = DecisionTreeClassifier(criterion='gini',
                                    max_depth=None,
                                    min_weight_fraction_leaf=0.001,
                                    splitter='best')
        dt.fit(X_train, y_train)
        self.dt = dt
        save_pickle(self.dt, 'ensemble/dt.pkl')
        # self.dt = load_pickle('twitterbot_pickles/dt.pkl')
        return dt

    def _majority(self, row):
        val = 1 if (row['rf'] + row['ab'] + row['gb'] + row['knn'] + row['nb']
                    + row['gnb'] + row['svc'] + row['svm'] + row['lr']
                    ) > 3 else 0
        return val

    def _tfidf_fit_transform(self, X):
        '''Fits and concatenates tf-idf columns to X for text, pos, and ner
        '''
        print('Calculating TF-IDF')
        # Perform TF-IDF on text column
        print('   on text column')
        self.tfidf_text = TfidfVectorizer(ngram_range=(1, 2),
                                          lowercase=False,
                                          token_pattern='\w+|\@\w+',
                                          norm='l2',
                                          max_df=.99,
                                          min_df=.01)
        tfidf_text = self.tfidf_text.fit_transform(X['text'])
        self.text_cols = self.tfidf_text.get_feature_names()
        idx = X.index
        tfidf_text = pd.DataFrame(tfidf_text.todense(),
                                  columns=[self.text_cols],
                                  index=idx)
        save_pickle(self.tfidf_text, 'ensemble/tfidf_text.pkl')
        # self.tfidf_text = load_pickle('twitterbot_pickles/tfidf_text.pkl')

        # Perform TF-IDF on ner column
        print('   on ner column')
        self.tfidf_ner = TfidfVectorizer(ngram_range=(1, 2),
                                         lowercase=False,
                                         norm='l2',
                                         max_df=.99,
                                         min_df=.01)
        tfidf_ner = self.tfidf_ner.fit_transform(X['ner'])
        self.ner_cols = self.tfidf_ner.get_feature_names()
        tfidf_ner = pd.DataFrame(tfidf_ner.todense(),
                                 columns=[self.ner_cols],
                                 index=idx)
        save_pickle(self.tfidf_ner, 'ensemble/tfidf_ner.pkl')
        # self.tfidf_ner = load_pickle('twitterbot_pickles/tfidf_ner.pkl')

        # Perform TF-IDF on pos column
        print('   on pos column')
        self.tfidf_pos = TfidfVectorizer(ngram_range=(2, 3),
                                         lowercase=False,
                                         norm='l2',
                                         max_df=.99,
                                         min_df=.01)
        tfidf_pos = self.tfidf_pos.fit_transform(X['pos'])
        self.pos_cols = self.tfidf_pos.get_feature_names()
        tfidf_pos = pd.DataFrame(tfidf_pos.todense(),
                                 columns=[self.pos_cols],
                                 index=idx)
        save_pickle(self.tfidf_pos, 'ensemble/tfidf_pos.pkl')
        # self.tfidf_pos = load_pickle('twitterbot_pickles/tfidf_pos.pkl')

        X = self._tfidf_remove_dups(X, tfidf_text, tfidf_pos, tfidf_ner)

        return X

    def _tfidf_transform(self, X):
        '''Performs a tf-idf transform on the given column of data
        '''
        X.reset_index(drop=True, inplace=True)
        tfidf_text = self.tfidf_text.transform(X['text'])
        tfidf_text = pd.DataFrame(tfidf_text.todense(),
                                  columns=[self.text_cols])

        tfidf_ner = self.tfidf_ner.transform(X['ner'])
        tfidf_ner = pd.DataFrame(tfidf_ner.todense(),
                                 columns=[self.ner_cols])

        tfidf_pos = self.tfidf_pos.transform(X['pos'])
        tfidf_pos = pd.DataFrame(tfidf_pos.todense(),
                                 columns=[self.pos_cols])

        X = self._tfidf_remove_dups(X, tfidf_text, tfidf_pos, tfidf_ner)

        return X

    def _tfidf_remove_dups(self, X, tfidf_text, tfidf_pos, tfidf_ner):
        '''Removes columns in tfidf_pos and tfidf_ner that are duplicates from
        tfidf_text, and concatentates the DataFrames
        '''
        # Drop ner columns also present in tfidf_text
        columns_to_keep = [x for x in tfidf_ner
                           if x not in tfidf_text]
        tfidf_ner = tfidf_ner[columns_to_keep]

        # Drop pos columns also present in ner
        columns_to_keep = [x for x in tfidf_pos
                           if x not in tfidf_ner]
        tfidf_pos = tfidf_pos[columns_to_keep]

        X = pd.concat([X, tfidf_text, tfidf_pos, tfidf_ner], axis=1)
        return X


def save_pickle(item, filename):
    # Save pickle file
    output = open(filename, 'wb')
    print('Pickle dump', filename)
    pickle.dump(item, output, protocol=4)
    output.close()


def load_pickle(filename):
    # Open pickle filename
    print('Pickle load', filename)
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    main()
