import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.load_data import load_json_list, apply_date_mask
from src.vader_sentiment import apply_vader
from src.style import apply_avg_lengths, tweet_length, punctuation_columns, \
                      quoted_retweet, apply_all_caps, mention_hashtag_url
from src.tweetstorm import tweetstorm
from src.time_of_day import time_of_day
from src.part_of_speech import pos_tagging, ner_tagging
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


def main():
    X_train, X_test, y_train, y_test = data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
    #X_train = feature_engineering(X_train)
    #tfidf_text_train = tf_idf_text(X_train)
    #tfidf_text_val = tf_idf_text(X_val)
    naive_bayes_accuracy = naive_bayes(np.array(X_train['text']),
                                       np.array(X_val['text']),
                                       np.array(y_train),
                                       np.array(y_val))

    print(naive_bayes_accuracy)


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

    # Look only at tweets between June 1, 2015 and March 26, 2017
    masked_df = apply_date_mask(raw_data, 'created_at',
                                '2015-06-01', '2017-03-26')
    df = masked_df.sort_values('created_at').reset_index(drop=True)

    # Look only at iPhone and Android tweets
#    df = df.loc[(df['source'] == 'Twitter for iPhone') |
#                (df['source'] == 'Twitter for Android')]

    # Dummify is_reply column
    df['in_reply_to_user_id_str'] = df['in_reply_to_user_id_str'].fillna(0)
    df['is_reply'] = np.where(df['in_reply_to_user_id_str'] == 0, False, True)

    # =========================================================================
    # Testing
    #df = df[0:30]
    # =========================================================================

    # Separate data and labels
    X = df.drop(['id_str', 'in_reply_to_user_id_str'], axis=1)
    y = pd.DataFrame(np.where(df['source'] == 'Twitter for Android', 1, 0))

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def feature_engineering(df):
    # =========================================================================
    # Feature engineering
    # =========================================================================

    # Create columns for vader sentiment
    df = apply_vader(df, 'text')

    # Create columns for average tweet, sentence, and word length of tweet
    df = tweet_length(df, 'text')
    df = apply_avg_lengths(df, 'text')

    # Create columns for counts of punctuation
    punctuation_dict = {'commas': ',', 'semicolons': ';', 'exclamations': '!',
                        'periods': '.', 'questions': '?', 'quote': '"'}

    df = punctuation_columns(df, 'text', punctuation_dict)

    # Create columns for counts of @mentions, #hashtags, and urls
    df = mention_hashtag_url(df, 'text')

    # Create column identifying if the tweet is surrounding by quote marks
    df = quoted_retweet(df, 'text')

    # Create column indicating the count of fully capitalized words in a tweet
    df = apply_all_caps(df, 'text')

    # Create column identifying if the tweet is part of a tweetstorm
    df = tweetstorm(df, 'text', 'source', 'created_at', 600)

    # Create column identifying the hour of the day that the tweet was posted
    df = time_of_day(df, 'created_at')

    # Part of speech tagging
    df['pos'] = df['text'].apply(pos_tagging)

    # Named Entity Recognition substitution
    df['ner'] = df['text'].apply(ner_tagging)

    return df


def tf_idf_text(df):
    # TF-IDF on raw text column
    tfidf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False,
                            token_pattern='\w+|\@\w+', norm='l2')
    tfidf_text = tfidf.fit_transform(df['text'])
    return tfidf_text


def tf_idf_pos(df):
    # TF-IDF on parts-of-speech tags
    tfidf = TfidfVectorizer(ngram_range=(2, 4), lowercase=False,
                            norm='l2')
    tfidf_pos = tfidf.fit_transform(df['pos'])

    return tfidf_pos


def naive_bayes(X_train, X_val, y_train, y_val):
    # TF-IDF on raw text column
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_val)
    accuracy = np.mean(predicted == y_val)

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                  lowercase=False,
                                                  token_pattern='\w+|\@\w+')),
                         ('tfidf', TfidfTransformer(norm='l2')),
                         ('clf', MultinomialNB()),
                         ])
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_val)
    accuracy = np.mean(predicted == y_val)

    '''
    tfidf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False,
                                         token_pattern='\w+|\@\w+', norm='l2')
    tfidf_train = tfidf.fit_transform(X_train['text'])
    tfidf_y_train = tfidf.fit_transform(y_train)
    tfidf_val = tfidf.fit_transform(X_val['text'])

    # run a simple Naive-Bayes and outut the accuracy
    clf = MultinomialNB().fit(tfidf_train, tfidf_y_train)
    predicted = clf.predict(tfidf_val)
    accuracy = np.mean(predicted == y_val)
    '''
    return accuracy


if __name__ == '__main__':
    main()
