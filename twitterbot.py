import json
import tweepy
import pandas as pd
import pickle
from TweetAuthorshipPredictor import TweetAuthorshipPredictor


credentials = json.load(open('.env/twitter_credentials.json'))

consumer_key = credentials['consumer_key']
consumer_secret = credentials['consumer_secret']
access_token = credentials['access_token']
access_token_secret = credentials['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

test = '14649582'
realDonaldTrump = '25073877'

# first_tweet(api)

# with open('trump.pkl', 'rb') as trump:
#     model = pickle.load(trump)

trumpstreamlistener = TrumpStreamListener()
trumpstream = tweepy.Stream(auth, trumpstreamlistener())

trumpstream.filter(follow=[realDonaldTrump])


class TrumpStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if status.author.id_str == realDonaldTrump:
            tweet_info = {'created_at': status.created_at,
                          'favorite_count': status.favorite_count,
                          'id_str': status.id_str,
                          'in_reply_to_user_id_str':
                          status.in_reply_to_user_id_str,
                          'is_retweet': status.retweeted,
                          'retweet_count': status.retweet_count,
                          'source': status.source,
                          'text': status.text}
            tweet = pd.DataFrame(tweet_info)
            prediction = predict_author(tweet)
            post_tweet(status, prediction)


def post_tweet(status, prediction):
    '''Takes a tweet, formats the response, and posts
    INPUT: string
    OUTPUT:
    '''
    url = str('https://twitter.com/realDonaldTrump/status/' + status.id_str)
    text = str(status.text)

    if prediction == 0:
        tweet = ('An aide probably wrote this: "{}..." {}'.
                 format(text[:140 - 32], url))
    else:
        tweet = ('Trump probably wrote this: "{}..." {}'.
                 format(text[:140 - 30 - len(url)], url))
    api.update_status(tweet)


def predict_author(tweet):
    return model.predict(tweet)


def first_tweet(api):
    api.update_with_media('images/trump_ticker.gif',
                          status="Stay tuned!...")


if __name__ == '__main__':
    main()
