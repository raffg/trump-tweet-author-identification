import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_vader_scores(text):
    '''
    takes a string of text and outputs four values for Vader's negative,
    neutral, positive, and compound sentiment scores
    INPUT: a string
    OUTPUT: a dictionary of four sentiment scores
    '''

    analyser = SentimentIntensityAnalyzer()
    return analyser.polarity_scores(text)


def apply_vader(df, column):
    '''
    takes a DataFrame with a specified column of text and adds four new columns
    to the DataFrame, corresponding to the Vader sentiment scores
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with four additional columns
    '''

    sentiment = pd.DataFrame(df[column].apply(get_vader_scores))
    unpacked = pd.DataFrame([d for idx, d in sentiment['text'].iteritems()],
                            index=sentiment.index)
    return pd.concat([df, unpacked], axis=1)
