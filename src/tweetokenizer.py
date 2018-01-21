from tweetokenize import Tokenizer


def tweet_tokens(tweet):
    '''
    Takes a tweet and replaces mentions, hashtags, urls, times, and numbers
    with a generic label
    INPUT: string
    OUTPUT: string
    '''

    gettokens = Tokenizer(usernames='USER', urls='URL',
                          hashtags='HASHTAG', times='TIME',
                          numbers='NUMBER', allcapskeep=True,
                          lowercase=False)
    tokens = gettokens.tokenize(tweet)
    tweet = ' '.join(tokens)

    return tweet


def tweet_tokenize(df, column):
    '''
    Takes a Data Frame and a specified column of tweets and creates a new
    column with the tweetokenized tweet
    INPUT: DateFrame, string
    OUTPUT: the original DataFrame with one new column
    '''

    new_df = df.copy()
    new_df['tweetokenize'] = new_df['text'].apply(tweet_tokens)
    return new_df
