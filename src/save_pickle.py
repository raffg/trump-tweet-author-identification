import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from src.load_data import load_json_list, apply_date_mask
from src.vader_sentiment import apply_vader
from src.style import apply_avg_lengths, tweet_length, punctuation_columns, \
                      quoted_retweet, apply_all_caps, mention_hashtag_url
from src.tweetstorm import tweetstorm
from src.time_of_day import time_of_day
from src.part_of_speech import pos_tagging, ner_tagging
from src.tweetokenizer import tweet_tokenize, tweet_tokens
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    # Create Train, Validation, and Test sets
    X, y = data('2015-06-01', '2017-03-26')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      random_state=1)

    # Apply feature engineering to all X sets
    print()
    print('Feature engineering on Train data')
    X_train = feature_engineering(X_train)
    print()
    print('Feature engineering on Validation data')
    X_val = feature_engineering(X_val)
    print()
    print('Feature engineering on Test data')
    X_test = feature_engineering(X_test)

    # Create ner column for Name Entity Recognition
    print()
    print('Performing NER on Train Data')
    X_train = named_entity_recognition(X_train)
    print('Performing NER on Validation Data')
    X_val = named_entity_recognition(X_val)
    print('Performing NER on Test Data')
    X_test = named_entity_recognition(X_test)

    # Create TF-IDF for NER column
    print()
    print('TF-IDF on ner column')
    tfidf_ner = TfidfVectorizer(ngram_range=(1, 2),
                                lowercase=False,
                                norm='l2',
                                min_df=0.01).fit(X_train['ner'])
    cols = tfidf_ner.get_feature_names()

    X_train_pos = tf_idf_matrix(X_train, 'ner', tfidf_ner, cols)
    X_val_pos = tf_idf_matrix(X_val, 'ner', tfidf_ner, cols)
    X_test_pos = tf_idf_matrix(X_test, 'ner', tfidf_ner, cols)

    # Create TF-IDF for NER_Tweetokenized column
    print()
    print('TF-IDF on ner_tweetokenized column')
    tfidf_ner_tweetokenized = TfidfVectorizer(ngram_range=(1, 2),
                                              lowercase=False,
                                              norm='l2',
                                              min_df=0.01).fit(
                                              X_train['ner_tweetokenized'])
    cols = tfidf_ner_tweetokenized.get_feature_names()

    X_train_pos = tf_idf_matrix(X_train, 'ner_tweetokenized',
                                tfidf_ner_tweetokenized, cols)
    X_val_pos = tf_idf_matrix(X_val, 'ner_tweetokenized',
                              tfidf_ner_tweetokenized, cols)
    X_test_pos = tf_idf_matrix(X_test, 'ner_tweetokenized',
                               tfidf_ner_tweetokenized, cols)

    # Create TF-IDF for text column
    print()
    print('TF-IDF on text column')
    tfidf_text = TfidfVectorizer(ngram_range=(1, 2),
                                 lowercase=False, token_pattern='\w+|\@\w+',
                                 norm='l2', min_df=0.01).fit(X_train['text'])
    cols = tfidf_text.get_feature_names()

    X_train_tfidf = tf_idf_matrix(X_train, 'text', tfidf_text, cols)
    X_val_tfidf = tf_idf_matrix(X_val, 'text', tfidf_text, cols)
    X_test_tfidf = tf_idf_matrix(X_test, 'text', tfidf_text, cols)

    # Create TF-IDF for pos column
    print()
    print('TF-IDF on pos column')
    tfidf_pos = TfidfVectorizer(ngram_range=(2, 3),
                                lowercase=False,
                                norm='l2',
                                min_df=0.01).fit(X_train['pos'])
    cols = tfidf_pos.get_feature_names()

    X_train_pos = tf_idf_matrix(X_train, 'pos', tfidf_pos, cols)
    X_val_pos = tf_idf_matrix(X_val, 'pos', tfidf_pos, cols)
    X_test_pos = tf_idf_matrix(X_test, 'pos', tfidf_pos, cols)

    # Save pickle file
    output = open('data_ner.pkl', 'wb')
    print()

    print('Pickle dump X_train')
    pickle.dump(X_train, output, protocol=4)
    print('Pickle dump X_val')
    pickle.dump(X_val, output, protocol=4)
    print('Pickle dump X_test')
    pickle.dump(X_test, output, protocol=4)

    print('Pickle dump X_train_tfidf')
    pickle.dump(X_train_tfidf, output, protocol=4)
    print('Pickle dump X_val_tfidf')
    pickle.dump(X_val_tfidf, output, protocol=4)
    print('Pickle dump X_test_tfidf')
    pickle.dump(X_test_tfidf, output, protocol=4)

    print('Pickle dump X_train_pos')
    pickle.dump(X_train_pos, output, protocol=4)
    print('Pickle dump X_val_pos')
    pickle.dump(X_val_pos, output, protocol=4)
    print('Pickle dump X_test_pos')
    pickle.dump(X_test_pos, output, protocol=4)

    print('Pickle dump y_train')
    pickle.dump(y_train, output, protocol=4)
    print('Pickle dump y_val')
    pickle.dump(y_val, output, protocol=4)
    print('Pickle dump y_test')
    pickle.dump(y_test, output, protocol=4)

    output.close()


def data(start_date, end_date):
    # =========================================================================
    # Load the data
    # =========================================================================
    print('Loading data')
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
    print('Masking data')
    masked_df = apply_date_mask(raw_data, 'created_at',
                                start_date, end_date)
    df = masked_df.sort_values('created_at').reset_index(drop=True)

    # =========================================================================
    # Testing
    # df = df[0:15]
    # =========================================================================

    # Look only at iPhone and Android tweets
#    df = df.loc[(df['source'] == 'Twitter for iPhone') |
#                (df['source'] == 'Twitter for Android')]

    # Dummify is_reply column
    print('Dummifying is_reply column')
    df['in_reply_to_user_id_str'].fillna(0, inplace=True)
    df['is_reply'] = np.where(df['in_reply_to_user_id_str'] == 0, False, True)

    # Separate data and labels
    print('Split data and labels')
    X = df.drop(['id_str', 'in_reply_to_user_id_str'], axis=1)
    y = pd.DataFrame(np.where(df['source'] == 'Twitter for Android', 1, 0))

    return X, y


def feature_engineering(df):
    # =========================================================================
    # Feature engineering
    # =========================================================================

    # Create columns for vader sentiment
    print('   calculating vader sentiment')
    df = apply_vader(df, 'text')

    # Create columns for average tweet, sentence, and word length of tweet
    print('   calculating average tweet, sentence, and word length')
    df = tweet_length(df, 'text')
    df = apply_avg_lengths(df, 'text')

    # Create columns for counts of punctuation
    print('   calculating punctuation counts')
    punctuation_dict = {'commas': ',', 'semicolons': ';', 'exclamations': '!',
                        'periods': '.', 'questions': '?', 'quote': '"'}

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

    # Create column of tweetokenized tweets
    print('   calculating tweetokenized tweets')
    df = tweet_tokenize(df, 'text')

    # Part of speech tagging
    print('   calculating part of speech')
    df['pos'] = df['tweetokenize'].apply(pos_tagging)

    return df


def named_entity_recognition(df):
    # Named Entity Recognition substitution
    print('   calculating named entity recognition')
    df['ner'] = df['text'].apply(ner_tagging)
    df['ner_tweetokenized'] = df['ner'].apply(tweet_tokens)
    return df


def tf_idf_matrix(df, column, vectorizer, cols):
    '''
    Takes a DataFrame, a column, a tfidfVectorizer, and a list of column names.
    Creates tf-idf matrix as a DataFrame
    INPUT: a DataFrame, string, list
    OUTPUT: a DataFrame
    '''

    print('   calculating TF-IDF matrix')
    matrix = vectorizer.transform(df[column])
    df_tfidf = pd.DataFrame(matrix.todense(), columns=[cols], index=df.index)
    # new_df = pd.concat([df, df_tfidf], axis=1)
    return df_tfidf


if __name__ == '__main__':
    main()
