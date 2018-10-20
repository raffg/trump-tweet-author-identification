import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import normalize
from sklearn import preprocessing


def get_vader_scores(text):
    '''
    Takes a string of text and outputs four values for Vader's negative,
    neutral, positive, and compound (normalized) sentiment scores
    INPUT: a string
    OUTPUT: a dictionary of four sentiment scores
    '''

    analyser = SentimentIntensityAnalyzer()
    return analyser.polarity_scores(text)


def apply_vader(df, column):
    '''
    Takes a DataFrame with a specified column of text and adds four new columns
    to the DataFrame, corresponding to the Vader sentiment scores
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with four additional columns
    '''

    sentiment = pd.DataFrame(df[column].apply(get_vader_scores))
    unpacked = pd.DataFrame([d for idx, d in sentiment[column].iteritems()],
                            index=sentiment.index)
    unpacked['compound'] += 1
    columns = {'neu': 'v_neutral', 'pos': 'v_positive', 'neg': 'v_negative'}
    unpacked.rename(columns=columns, inplace=True)
    return pd.concat([df, unpacked], axis=1)
