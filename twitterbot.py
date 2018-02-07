import json
import tweepy
import pickle


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


def compose_tweet(tweet, prediction):
    '''Takes a tweet and formats a response
    INPUT: string
    OUTPUT:
    '''
    url = str(tweet[url])
    text = str(tweet[text])
    if prediction == 0:
        text = ('Trump probably did not write this. "{}..." {}'.
                format(text[:140 - 38 - len(url)], url))
    else:
        text = ('Trump probably wrote this. "{}..." {}'.
                format(text[:140 - 30 - len(url)], url))
    return text


def post_tweet(api, text):
    api.update_status(text)


def predict_author(model, tweet):
    return model.predict(tweet)


if __name__ == '__main__':
    main()
