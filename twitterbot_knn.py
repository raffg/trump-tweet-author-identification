import json
import tweepy
import pandas as pd
import pickle
from TweetAuthorshipPredictor import TweetAuthorshipPredictor
from src.feature_pipeline import feature_pipeline


credentials = json.load(open('.env/twitter_credentials.json'))

consumer_key = credentials['consumer_key']
consumer_secret = credentials['consumer_secret']
access_token = credentials['access_token']
access_token_secret = credentials['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

realDonaldTrump = '25073877'

with open('pickle/ensemble_knn.pkl', 'rb') as trump:
    model = pickle.load(trump)

std = ['favorite_count', 'retweet_count', 'compound', 'anger',
       'anticipation', 'disgust', 'fear', 'joy', 'negative',
       'positive', 'sadness', 'surprise', 'trust',
       'tweet_length', 'avg_sentence_length', 'avg_word_length',
       'commas', 'semicolons', 'exclamations', 'periods',
       'questions', 'quotes', 'ellipses', 'mentions', 'hashtags',
       'urls', 'all_caps', 'hour']

feat = ['created_at', 'favorite_count', 'is_retweet',
        'retweet_count',
        'text', 'is_reply', 'compound',
        'v_negative', 'v_neutral', 'v_positive', 'anger',
        'anticipation', 'disgust', 'fear', 'joy', 'negative',
        'positive', 'sadness', 'surprise', 'trust',
        'tweet_length', 'avg_sentence_length', 'avg_word_length',
        'commas', 'semicolons', 'exclamations', 'periods',
        'questions', 'quotes', 'ellipses', 'mentions', 'hashtags',
        'urls', 'is_quoted_retweet', 'all_caps', 'tweetstorm',
        'hour', 'hour_20_02', 'hour_14_20', 'hour_08_14',
        'hour_02_08', 'start_mention', 'ner', 'pos']


def load_pickle(filename):
    # Open pickle filename
    print('Pickle load', filename)
    with open(filename, 'rb') as f:
        return pickle.load(f)


knn_pca = load_pickle('ensemble/knn_pca.pkl')
tfidf_pos = load_pickle('tfidf_pos.pkl')
tfidf_ner = load_pickle('tfidf_ner.pkl')
tfidf_text = load_pickle('tfidf_text.pkl')
text_cols = tfidf_text.get_feature_names()
ner_cols = tfidf_ner.get_feature_names()
pos_cols = tfidf_pos.get_feature_names()
scaler = load_pickle('ensemble/scaler.pkl')
top_feats = load_pickle('ensemble/top_feats.pkl')


class TrumpStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if status.author.id_str == realDonaldTrump:
            # tweet_info = {'created_at': status.created_at,
            #               'favorite_count': status.favorite_count,
            #               'id_str': status.id_str,
            #               'in_reply_to_user_id_str':
            #               status.in_reply_to_user_id_str,
            #               'is_retweet': status.retweeted,
            #               'retweet_count': status.retweet_count,
            #               'source': status.source,
            #               'text': status.text}
            tweet = df = pd.DataFrame(columns=['created_at',
                                               'favorite_count',
                                               'id_str',
                                               'in_reply_to_user_id_str',
                                               'is_retweet',
                                               'retweet_count',
                                               'source',
                                               'text'])

            tweet.loc[0] = [status.created_at,
                            status.favorite_count,
                            status.id_str,
                            status.in_reply_to_user_id_str,
                            status.retweeted,
                            status.retweet_count,
                            status.source,
                            status.text]
            prediction = predict_author(tweet)
            post_tweet(status, prediction)


def post_tweet(status, prediction):
    '''Takes a tweet, formats the response, and posts to Twitter
    INPUT: string
    OUTPUT:
    '''
    url = str('https://twitter.com/realDonaldTrump/status/' + status.id_str)
    url = ('https://twitter.com/' + status.user.screen_name +
           '/status/' + status.id_str)
    text = str(status.text)

    if prediction == 0:
        tweet = ('An aide probably wrote this: "{}..." {}'.
                 format(text[:140 - 32], url))
    else:
        tweet = ('Trump probably wrote this: "{}..." {}'.
                 format(text[:140 - 30 - len(url)], url))
    api.update_status(tweet)
    print(tweet)


def predict_author(tweet):
    X, X_std = prepare_data_for_predict(tweet)
    X_knn = knn_pca.transform(X_std[top_feats[:13]])
    return model.predict(X_knn)


def first_tweet(api):
    api.update_with_media('images/trump_ticker.gif',
                          status="Stay tuned!...")


def prepare_data_for_predict(X):
    ''' Processes the X data with all features and standardizes.
    '''
    # Create new feature columns
    X = feature_pipeline(X)
    X = tfidf_transform(X[feat])
    X_std = standardize(X)
    return X, X_std


def tfidf_transform(X):
    '''Performs a tf-idf transform on the given column of data
    '''
    X.reset_index(drop=True, inplace=True)
    _tfidf_text = tfidf_text.transform(X['text'])
    _tfidf_text = pd.DataFrame(_tfidf_text.todense(),
                               columns=[text_cols])

    _tfidf_ner = tfidf_ner.transform(X['ner'])
    _tfidf_ner = pd.DataFrame(_tfidf_ner.todense(),
                              columns=[ner_cols])

    _tfidf_pos = tfidf_pos.transform(X['pos'])
    _tfidf_pos = pd.DataFrame(_tfidf_pos.todense(),
                              columns=[pos_cols])

    X = tfidf_remove_dups(X, _tfidf_text, _tfidf_pos, _tfidf_ner)

    return X


def tfidf_remove_dups(X, tfidf_text, tfidf_pos, tfidf_ner):
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


def standardize(X):
    print('Performing Standardization')
    X_std = X.copy()
    cols = X[std].columns
    X_std[std] = pd.DataFrame(scaler.transform(
                              X[std]),
                              index=X.index,
                              columns=cols)
    return X_std


trumpstreamlistener = TrumpStreamListener()
trumpstream = tweepy.Stream(auth, trumpstreamlistener)

trumpstream.filter(follow=[realDonaldTrump])
