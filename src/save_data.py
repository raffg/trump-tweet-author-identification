import pandas as pd
import numpy as np
import pickle
from src.load_pickle import load_pickle
from src.load_data import load_json_list, apply_date_mask
from src.vader_sentiment import apply_vader
from src.text_emotion import text_emotion
from src.style import apply_avg_lengths, tweet_length, punctuation_columns, \
                      quoted_retweet, apply_all_caps, mention_hashtag_url, \
                      mention_start
from src.tweetstorm import tweetstorm
from src.time_of_day import time_of_day, period_of_day
from src.part_of_speech import pos_tagging, ner_tagging
from src.tweetokenizer import tweet_tokenize, tweet_tokens
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def main():
    df = data()

    df.to_pickle('pickle/all_data_raw.pkl')

    # =========================================================================
    # Testing
    # df = df[0:15]
    # =========================================================================

    # Apply feature engineering
    df = feature_engineering(df)
    df.to_pickle('pickle/all_data_features.pkl')

    # Standardize
    feat = ['favorite_count', 'retweet_count', 'compound', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'all_caps', 'hour']
    df_std = standardize(df, feat)
    df_std.to_pickle('pickle/all_data_features_std.pkl')

    # Separate data into known/unknown author time periods and campaign only
    df_feature_data_labeled = apply_date_mask(df, 'created_at',
                                              '2009-01-01', '2017-03-26')
    df_feature_data_unlabeled = apply_date_mask(df, 'created_at',
                                                '2017-03-26', '2018-01-01')
    df_feature_data_labeled_std = apply_date_mask(df_std, 'created_at',
                                                  '2009-01-01', '2017-03-26')
    df_feature_data_unlabeled_std = apply_date_mask(df_std, 'created_at',
                                                    '2017-03-26', '2018-01-01')
    df_feature_data_campaign = apply_date_mask(df, 'created_at',
                                               '2015-06-01', '2017-03-26')
    df_feature_data_campaign_std = apply_date_mask(df_std, 'created_at',
                                                   '2015-06-01', '2017-03-26')

    df_dict = {'feature_data_labeled': df_feature_data_labeled,
               'feature_data_unlabeled': df_feature_data_unlabeled,
               'feature_data_labeled_std': df_feature_data_labeled_std,
               'feature_data_unlabeled_std': df_feature_data_unlabeled_std,
               'feature_data_campaign': df_feature_data_campaign,
               'feature_data_campaign_std': df_feature_data_campaign_std}

    for df in list(df_dict):
        df_dict[df].to_pickle('pickle/{}.pkl'.format(df))

    # Process and save all TF-IDF matrices
    (all_data_labeled,
     all_data_unlabled) = tfidf_process((df_feature_data_labeled,
                                        df_feature_data_unlabeled))
    (all_data_labeled_std,
     all_data_unlabled_std) = tfidf_process((df_feature_data_labeled_std,
                                            df_feature_data_unlabeled_std))
    (all_data_campaign,
     all_data_campaign_y) = tfidf_process((df_feature_data_campaign,
                                          df_feature_data_unlabeled))
    (all_data_campaign_std,
     all_data_campaign_y_std) = tfidf_process((df_feature_data_campaign_std,
                                              df_feature_data_unlabeled_std))

    df_dict = {'all_data_labeled': all_data_labeled,
               'all_data_unlabeled': all_data_unlabeled,
               'all_labeled_std': all_labeled_std,
               'all_unlabeled_std': all_unlabeled_std,
               'all_data_campaign': all_data_campaign,
               'all_data_campaign_std': all_data_campaign_std}

    for df in list(df_dict):
        df_dict[df].to_pickle('pickle/{}.pkl'.format(df))

    # Perform train/test splits on all data sets and save pickles
    (train_all, test_all,
     y_train_all, y_test_all) = train_test(all_data_labeled)
    _train_all = train_all.copy()
    (train_all, test_all) = tfidf_process((train_all, test_all))

    (train_all_std, test_all_std,
     y_train_all_std, y_test_all_std) = train_test(all_data_labeled_std)
    _train_all_std = train_all.copy()
    (train_all_std, test_all_std) = tfidf_process((train_all_std,
                                                  test_all_std))

    (train_campaign,
     test_campaign,
     y_train_campaign,
     y_test_campaign) = train_test(all_data_campaign)
    _train_campaign = train_all.copy()
    (train_campaign,
     test_campaign) = tfidf_process((train_campaign, test_campaign))

    (train_campaign_std,
     test_campaign_std,
     y_train_campaign_std,
     y_test_campaign_std) = train_test(all_data_campaign_std)
    _train_campaign_std = train_all.copy()
    (train_campaign_std,
     test_campaign_std) = tfidf_process((train_campaign_std,
                                        test_campaign_std))

    df_dict = {'train_all': train_all,
               'test_all': test_all,
               'y_train_all': y_train_all,
               'y_test_all': y_test_all,
               'train_all_std': train_all_std,
               'test_all_std': test_all_std,
               'y_train_all_std': y_train_all_std,
               'y_test_all_std': y_test_all_std,
               'train_campaign': train_campaign,
               'test_campaign': test_campaign,
               'y_train_campaign': y_train_campaign,
               'y_test_campaign': y_test_campaign,
               'train_campaign_std': train_campaign_std,
               'test_campaign_std': test_campaign_std,
               'y_train_campaign_std': y_train_campaign_std,
               'y_test_campaign_std': y_test_campaign_std}

    for df in list(df_dict):
        df_dict[df].to_pickle('pickle/{}.pkl'.format(df))

    # Perform train/val/test splits on all data sets and save pickles
    (train_all, val_all,
     y_train_all, y_val_all) = train_val(_train_all, y_train_all)
    (train_all, val_all) = tfidf_process((train_all, val_all))

    (train_all_std, val_all_std,
     y_train_all_std, y_val_all_std) = train_val(_train_all_std,
                                                 y_train_all_std)
    (train_all_std, val_all_std) = tfidf_process((train_all_std, val_all_std))

    (train_campaign,
     val_campaign,
     y_train_campaign,
     y_val_campaign) = train_val(_train_campaign, y_train_campaign)
    (train_campaign,
     val_campaign) = tfidf_process((train_campaign, val_campaign))

    (train_campaign_std,
     val_campaign_std,
     y_train_campaign_std,
     y_val_campaign_std) = train_val(_train_campaign_std, y_train_campaign_std)
    (train_campaign_std,
     val_campaign_std) = tfidf_process((train_campaign_std, val_campaign_std))

    df_dict = {'train_val_all': train_all,
               'val_all': val_all,
               'y_train_val_all': y_train_all,
               'y_val_all': y_val_all,
               'train_val_all_std': train_all_std,
               'val_all_std': val_all_std,
               'y_train_val_all_std': y_train_all_std,
               'y_val_all_std': y_val_all_std,
               'train_val_campaign': train_campaign,
               'val_campaign': val_campaign,
               'y_train_val_campaign': y_train_campaign,
               'y_val_campaign': y_val_campaign,
               'train_val_campaign_std': train_campaign_std,
               'val_campaign_std': val_campaign_std,
               'y_train_val_campaign_std': y_train_campaign_std,
               'y_val_campaign_std': y_val_campaign_std}

    for df in list(df_dict):
        df_dict[df].to_pickle('pickle/{}.pkl'.format(df))


def train_test(df):
    '''
    Takes a DataFrame of labeled data, creates a y-labeled DataFrame from
    the 'source' column, and performs train-test split. Creates all TF-IDF
    matrices on test and train data. Outputs four DataFrames for X_train,
    X_test, y_train, and y_test
    '''

    y = pd.DataFrame(np.where(df['source'] == 'Twitter for Android', 1, 0))
    X = df.drop(['source'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)

    return X_train, X_test, y_train, y_test


def train_val(df, y):
    '''
    Takes a DataFrame of labeled data, and a y-labeled DataFrame from
    the 'source' column, and performs train-test split. Creates all TF-IDF
    matrices on test and train data. Outputs four DataFrames for X_train,
    X_test, y_train, and y_test
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)

    return X_train, X_test, y_train, y_test


def tfidf_data(df, column, vectorizer):
    '''
    Takes a tuple of two DataFrames, train and test, and a column of text;
    trains a TF-IDF vectorizer on the text column of the first DataFrame and
    then applies it to both. Returns
    TF-IDF DataFrames for both.
    '''

    print()
    print('Performing TF-IDF')
    tfidf = vectorizer.fit(df[0][column])
    cols = tfidf.get_feature_names()
    idx0 = df[0].index
    idx1 = df[1].index
    print(df[0])
    print('=============================================================')
    print(df[1])

    train = vectorizer.transform(df[0][column])
    df_train = pd.DataFrame(train.todense(), columns=[cols], index=idx0)

    test = vectorizer.transform(df[1][column])
    df_test = pd.DataFrame(test.todense(), columns=[cols], index=idx1)

    return df_train, df_test


def tfidf_process(pair):
    '''
    Takes a pair of Train/Test DataFrames and performs TF-IDF on the text, ner,
    and pos columns. Outputs the new train and test DataFrames.
    '''

    print()
    print('Calculating TF-IDF')
    # Perform TF-IDF on text column
    tfidf_text = TfidfVectorizer(ngram_range=(1, 2),
                                 lowercase=False,
                                 token_pattern='\w+|\@\w+',
                                 norm='l2',
                                 max_df=.99,
                                 min_df=.01)
    (tfidf_labeled_text,
     tfidf_unlabled_text) = tfidf_data(pair,
                                       'text',
                                       tfidf_text)

    # Perform TF-IDF on ner column
    tfidf_ner = TfidfVectorizer(ngram_range=(1, 2),
                                lowercase=False,
                                norm='l2',
                                max_df=.99,
                                min_df=.01)
    (tfidf_labeled_ner,
     tfidf_unlabled_ner) = tfidf_data(pair,
                                      'ner',
                                      tfidf_ner)

    # Perform TF-IDF on pos column
    tfidf_pos = TfidfVectorizer(ngram_range=(2, 3),
                                lowercase=False,
                                norm='l2',
                                max_df=.99,
                                min_df=.01)
    (tfidf_labeled_pos,
     tfidf_unlabled_pos) = tfidf_data(pair,
                                      'pos',
                                      tfidf_pos)

    # Drop ner columns also present in tfidf_text
    columns_to_keep = [x for x in tfidf_labeled_ner
                       if x not in tfidf_labeled_text]
    tfidf_labeled_ner = tfidf_labeled_ner[columns_to_keep]
    tfidf_unlabeled_ner = tfidf_unlabeled_ner[columns_to_keep]

    # Drop pos columns also present in ner
    columns_to_drop = ['LOCATION LOCATION',
                       'ORGANIZATION ORGANIZATION',
                       'PERSON PERSON']
    tfidf_labeled_pos = tfidf_labeled_pos.drop(columns_to_drop, axis=1)
    tfidf_unlabeled_pos = tfidf_unlabeled_pos.drop(columns_to_drop, axis=1)

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention', 'source']

    all_data_labeled = pd.concat([pair[0][feat], tfidf_labeled_text,
                                 tfidf_labeled_pos, tfidf_labeled_ner],
                                 axis=1)
    all_data_unlabeled = pd.concat([pair[1][feat], tfidf_unlabeled_text,
                                   tfidf_unlabeled_pos, tfidf_unlabeled_ner],
                                   axis=1)

    return all_data_labeled, all_data_unlabeled


def data():
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

    df = load_json_list(data_list)

    df = df.sort_values('created_at').reset_index(drop=True)

    # Remove 3 tweets from 2018
    df = df[:-3]

    # Dummify is_reply column
    print('Dummifying is_reply column')
    df['in_reply_to_user_id_str'].fillna(0, inplace=True)
    df['is_reply'] = np.where(np.isnan(
                              df['in_reply_to_user_id_str']), 0, 1)

    return df


def feature_engineering(df):
    # =========================================================================
    # Feature engineering
    # =========================================================================
    print()
    print('Feature engineering')

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

    # =========================================================================
    '''
    # Create ner column for Name Entity Recognition
    print()
    print('Performing NER')
    df = named_entity_recognition(df)
    '''

    # Recreate ner data
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle('pickle/vault/all_data_old.pkl')

    temp_data = pd.concat([X_train, X_val, X_test], axis=0)
    temp_data.sort_values(by=['created_at'], inplace=True)

    df['ner'] = temp_data['ner'].reset_index(drop=True)
    # =========================================================================

    return df


def named_entity_recognition(df):
    # Named Entity Recognition substitution
    print('   calculating named entity recognition')
    df['ner'] = df['tweetokenize'].apply(ner_tagging)
    return df


def standardize(df, feature_list):
    '''
    Takes DataFrame and a list of numerical features to standardize, and
    standardizes the feature columns. Outputs the original DataFrame with
    features in feature_list standardized and other features untouched.
    INPUT:  DataFrame, list
    OUTPUT: DataFrame
    '''

    print()
    print('Standardizing data')
    scaler = StandardScaler()

    new_df = df.copy()
    cols = df[feature_list].columns

    scaler.fit(new_df[feature_list])
    new_df[feature_list] = pd.DataFrame(scaler.transform(
                                        new_df[feature_list]),
                                        index=df.index, columns=cols)
    return new_df


if __name__ == '__main__':
    main()
