import json
import tweepy
import pickle
from TweetAuthorshipPredictor import TweetAuthorshipPredictor


def main():
    credentials = json.load(open('.env/twitter_credentials.json'))

    consumer_key = credentials['consumer_key']
    consumer_secret = credentials['consumer_secret']
    access_token = credentials['access_token']
    access_token_secret = credentials['access_token_secret']

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    with open('trump.pkl', 'rb') as trump:
        model = pickle.load(trump)


def post_tweet(tweet, prediction):
    '''Takes a tweet and formats a response
    INPUT: string
    OUTPUT:
    '''
    url = str(tweet[url])
    text = str(tweet[text])
    status_id = tweet[status_id]

    if prediction == 0:
        text = ('Trump probably did not write this: "{}..." {}'.
                format(text[:140 - 38 - len(url)], url))
        image = 'images/not_trump.png'
    else:
        text = ('Trump probably wrote this: "{}..." {}'.
                format(text[:140 - 30 - len(url)], url))
        image = 'images/trump.png'

    api.update_with_media(image, status=text, in_reply_to_status_id=status_id)


def post_tweet(api, text):
    api.update_with_media(text)


def predict_author(model, tweet):
    return model.predict(tweet)


if __name__ == '__main__':
    main()
