from tweetokenize import Tokenizer


def tweet_tokens(tweet):
    '''
    Takes a tweet and replaces mentions, hashtags, urls, times, and numbers
    with a generic label
    INPUT: string
    OUTPUT: string
    '''

    gettokens = Tokenizer(usernames='<USER>', urls='<URL>',
                          hashtags='<HASHTAG>', times=',<TIME>',
                          numbers='<NUMBER>', allcapskeep=True,)
    tokens = gettokens.tokenize(tweet)
    return tokens
