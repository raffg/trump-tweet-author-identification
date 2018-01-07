import pandas as pd
import preprocessor as p
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.load_data import load_json_list, apply_date_mask, sort_by_date
from src.vader_sentiment import apply_vader
from src.style import apply_avg_lengths, tweet_length, punctuation_columns, \
                      quoted_retweet, apply_all_caps
from src.tweetstorm import tweetstorm
from src.time_of_day import time_of_day
from src.part_of_speech import pos_tagging, ner_tagging


def main():
    df = data()
    df = feature_engineering(df)
    tfidf_matrix = tf_idf(df)

    print(df)


def data():
    # =========================================================================
    # Load the data
    # =========================================================================
    data_list = (['data/condensed_2009.json',
                  'data/condensed_2010.json',
                  'data/condensed_2011.json',
                  'data/condensed_2012.json',
                  'data/condensed_2013.json',
                  'data/condensed_2014.json',
                  'data/condensed_2015.json',
                  'data/condensed_2016.json',
                  'data/condensed_2017.json'])

    raw_data = load_json_list(data_list)
    masked_df = apply_date_mask(raw_data, 'created_at',
                                '2015-06-01', '2017-03-26')
    df = sort_by_date(masked_df, 'created_at')

    # =========================================================================
    # Testing
    df = df[0:10]
    # =========================================================================

    return df


def feature_engineering(df):
    # =========================================================================
    # Feature engineering
    # =========================================================================

    # Create columns for vader sentiment
    df = apply_vader(df, 'text')

    # Create columns for average tweet, sentence, and word length of tweet
    df = tweet_length(df, 'text')
    df = apply_avg_lengths(df, 'text')

    # Create columns for counts of @mentions, #hashtags, urls, and punctuation
    punctuation_dict = {'mentions': '@', 'hashtags': '#', 'urls': '://',
                        'commas': ',', 'semicolons': ';', 'exclamations': '!',
                        'periods': '.', 'questions': '?', 'quote': '"'}

    df = punctuation_columns(df, 'text', punctuation_dict)

    # Create column identifying if the tweet is surrounding by quote marks
    df = quoted_retweet(df, 'text')

    # Create column indicating the count of fully capitalized words in a tweet
    df = apply_all_caps(df, 'text')

    # Create column identifying if the tweet is part of a tweetstorm
    df = tweetstorm(df, 'text', 'source', 'created_at', 600)

    # Create column identifying the hour of the day that the tweet was posted
    df = time_of_day(df, 'created_at')

    # Uniformize @mentions, #hashtags, and URLs
    df['tweetokenize'] = df['text'].apply(p.tokenize)

    # Part of speech tagging
    df['pos'] = df['text'].apply(pos_tagging)

    # Named Entity Recognition substitution
    df['ner'] = df['text'].apply(ner_tagging)

    return df


def tf_idf(df):
    # =========================================================================
    # TF-IDF
    # =========================================================================

    tfidf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False,
                            token_pattern='\w+|\@\w+', norm='l2')
    tfidf_matrix = tfidf.fit_transform(df['text'])

    return tfidf_matrix


if __name__ == '__main__':
    main()
