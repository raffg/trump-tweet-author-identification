import pandas as pd
import numpy as np
from src.vader_sentiment import apply_vader
from src.text_emotion import text_emotion
from src.style import apply_avg_lengths, tweet_length, punctuation_columns, \
                      quoted_retweet, apply_all_caps, mention_hashtag_url, \
                      mention_start
from src.tweetstorm import tweetstorm
from src.time_of_day import time_of_day, period_of_day
from src.part_of_speech import pos_tagging, ner_tagging
from src.tweetokenizer import tweet_tokenize, tweet_tokens


def feature_pipeline(df):
    # =========================================================================
    # Feature engineering
    # =========================================================================
    print()
    print('Feature engineering')

    # Dummify is_reply column
    print('Dummifying is_reply column')
    df['in_reply_to_user_id_str'].fillna(0, inplace=True)
    df['is_reply'] = np.where(df['in_reply_to_user_id_str'], 1, 0)

    # Create columns for vader sentiment
    print('   calculating vader sentiment')
    df = apply_vader(df, 'text')

    # Create columns for NRC Emotion Lexicon
    print('   calculating NRC Emotion Lexicon score')
    df = text_emotion(df, 'text')

    # Create columns for average tweet, sentence, and word length of tweet
    print('   calculating average tweet, sentence, and word length')
    df = tweet_length(df, 'text')
    df = apply_avg_lengths(df, 'text')

    # Create columns for counts of punctuation
    print('   calculating punctuation counts')
    punctuation_dict = {'commas': ',', 'semicolons': ';', 'exclamations': '!',
                        'periods': '.', 'questions': '?', 'quotes': '"',
                        'ellipses': '...'}

    df = punctuation_columns(df, 'text', punctuation_dict)

    # Create columns for counts of @mentions, #hashtags, and urls
    print('   calculating mentions, hashtags, and url counts')
    df = mention_hashtag_url(df, 'text')

    # Create column identifying if the tweet is surrounding by quote marks
    print('   calculating quoted retweet')
    df = quoted_retweet(df, 'text')

    # Create column indicating the count of fully capitalized words in a tweet
    print('   calculating fully capitalized word counts')
    df = apply_all_caps(df, 'text')

    # Create column identifying if the tweet is part of a tweetstorm
    print('   calculating tweetstorm')
    df = tweetstorm(df, 'text', 'source', 'created_at', 600)

    # Create column identifying the hour of the day that the tweet was posted
    print('   calculating time of day')
    df = time_of_day(df, 'created_at')

    # Create column identifying the period of the day, in 6-hour increments
    print('   calculating period of day')
    df = period_of_day(df, 'created_at')

    # Create column of tweetokenize tweets
    print('   calculating tweetokenize tweets')
    df = tweet_tokenize(df, 'text')

    # Create column identifying if the tweet begins with an @mentions
    print('   calculating @mention beginnings')
    df['start_mention'] = df['tweetokenize'].apply(mention_start)

    # Part of speech tagging
    print('   calculating part of speech')
    df['pos'] = df['tweetokenize'].apply(pos_tagging)

    # Create ner column for Name Entity Recognition
    print()
    print('Performing NER')
    df['ner'] = df['tweetokenize'].apply(ner_tagging)

    return df.drop(['source'], axis=1)
