import json
import tweepy
import pandas as pd
import pickle
from time import sleep
from TweetAuthorshipPredictor import TweetAuthorshipPredictor


credentials = json.load(open('.env/twitter_credentials.json'))

consumer_key = credentials['consumer_key']
consumer_secret = credentials['consumer_secret']
access_token = credentials['access_token']
access_token_secret = credentials['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

realDonaldTrump = '25073877'
# realDonaldTrump = '14649582'  # test

with open('twitterbot_pickles/trump.pkl', 'rb') as trump:
    print('Loading model...')
    model = pickle.load(trump)


class TrumpStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if status.author.id_str == realDonaldTrump:
            tweet = pd.DataFrame(columns=['created_at',
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

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            print('Hit rate limit, pausing 60 seconds')
            sleep(60)
            return True


def post_tweet(status, prediction):
    '''Takes a tweet, formats the response, and posts to Twitter
    INPUT: string
    OUTPUT:
    '''
    url = ('https://twitter.com/' + status.user.screen_name +
           '/status/' + status.id_str)
    text = str(status.text)
    if len(text) >= 114:
        text = text[:114] + '…'

    proba = .99 if prediction[1] > .99 else prediction[1]

    if prediction[0] == 0:
        tweet = ('I am {0:.0%} confident an aide wrote this:\n'
                 '"{1}"\n'
                 '@realDonaldTrump {2}'.
                 format((1 - proba), text, url))
    else:
        tweet = ('I am {0:.0%} confident Trump wrote this:\n'
                 '"{1}"\n'
                 '@realDonaldTrump {2}'.
                 format(proba, text, url))
    print(tweet)
    print()
    api.update_status(tweet)


def predict_author(tweet):
    return model.predict(tweet)


def first_tweet(api):
    api.update_with_media('images/trump_ticker.gif',
                          status="Stay tuned!...")


def start_stream():
    while True:
        try:
            trumpstream = tweepy.Stream(auth, trumpstreamlistener)
            trumpstream.filter(follow=[realDonaldTrump])
        except:
            continue


trumpstreamlistener = TrumpStreamListener()
print('Ready!')
start_stream()
