import pandas as pd
import numpy as np
from src.vader_sentiment import apply_vader
from src.text_emotion import text_emotion
from src.style import apply_avg_lengths, tweet_length, punctuation_columns, \
                      quoted_retweet, apply_all_caps, mention_hashtag_url, \
                      mention_start, random_capitalization
from src.tweetstorm import tweetstorm
from src.time_of_day import time_of_day, period_of_day, day_of_week, weekend
from src.part_of_speech import pos_tagging, ner_tagging
from src.tweetokenizer import tweet_tokenize, tweet_tokens


def feature_pipeline(df, verbose=False):
    # =========================================================================
    # Feature engineering
    # =========================================================================
    if verbose:
        print()
        print('Feature engineering')

    # Dummify is_reply column
    if verbose:
        print('   dummifying is_reply column')
    df['in_reply_to_user_id_str'].fillna(0, inplace=True)
    df['is_reply'] = np.where(df['in_reply_to_user_id_str'], 1, 0)

    # Create columns for vader sentiment
    if verbose:
        print('   calculating vader sentiment')
    df = apply_vader(df, 'text')

    # Create columns for NRC Emotion Lexicon
    if verbose:
        print('   calculating NRC Emotion Lexicon score')
    df = text_emotion(df, 'text')

    # Create columns for average tweet, sentence, and word length of tweet
    if verbose:
        print('   calculating average sentence and word length')
    df = apply_avg_lengths(df, 'text')

    # Create columns for counts of punctuation
    if verbose:
        print('   calculating punctuation counts')
    punctuation_dict = {'commas': ',', 'semicolons': ';', 'exclamations': '!',
                        'periods': '.', 'questions': '?', 'quotes': '"',
                        'ellipses': '...'}

    df = punctuation_columns(df, 'text', punctuation_dict)

    # Create columns for counts of @mentions, #hashtags, and urls
    if verbose:
        print('   calculating mentions, hashtags, and url counts')
    df = mention_hashtag_url(df, 'text')

    # Create column identifying if the tweet is surrounding by quote marks
    if verbose:
        print('   calculating quoted retweet')
    df = quoted_retweet(df, 'text')

    # Create column indicating the count of fully capitalized words in a tweet
    if verbose:
        print('   calculating fully capitalized word counts')
    df = apply_all_caps(df, 'text')

    # Create column identifying if the tweet is part of a tweetstorm
    # if verbose:
    #     print('   calculating tweetstorm')
    # df = tweetstorm(df, 'text', 'source', 'created_at', 600)

    # Create column identifying the hour of the day that the tweet was posted
    if verbose:
        print('   calculating time of day')
    df = time_of_day(df, 'created_at')

    # Create column identifying the day of the week that the tweet was posted
    if verbose:
        print('   calculating day of week')
    df = day_of_week(df, 'created_at')

    # Create column identifying if the day of the week occurred on a weekend
    if verbose:
        print('   calculating weekend')
    df = weekend(df, 'day_of_week')

    # Create column identifying the period of the day, in 6-hour increments
    if verbose:
        print('   calculating period of day')
    df = period_of_day(df, 'created_at')

    # Create column finding the number of randomly capitalized words
    if verbose:
        print('   calculating randomly capitalized words')
    df = random_capitalization(df, 'text')

    # Create column of tweetokenize tweets
    if verbose:
        print('   calculating tweetokenize tweets')
    df = tweet_tokenize(df, 'text')

    # Create column identifying if the tweet begins with an @mentions
    if verbose:
        print('   calculating @mention beginnings')
    df['start_mention'] = df['tweetokenize'].apply(mention_start)

    # Part of speech tagging
    if verbose:
        print('   calculating part of speech')
    df['pos'] = df['tweetokenize'].apply(pos_tagging)

    # Create ner column for Name Entity Recognition
    if verbose:
        print()
        print('Performing NER')
    df['ner'] = df['tweetokenize'].apply(ner_tagging)

    return df.drop(['source'], axis=1)
