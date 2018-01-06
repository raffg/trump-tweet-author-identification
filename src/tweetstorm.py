import pandas as pd


def tweetstorm(df, tweet, source, timestamp, time_threshold):
    '''
    Takes a DataFrame with a specified column containing tweets, a specified
    column identifying the source of the tweet, a specified column indicating
    the timestamp of the tweet, and a threshold in seconds defining the
    maximimum time which can pass between tweets to define a tweetstorm
    INPUT: DataFrame, string, string, string, int
    OUTPUT: the original DataFrame with one new column
    '''

    temp = pd.DataFrame()
    df = df.copy()
    temp['time_diff'] = df.groupby(source)[timestamp].diff().dt.total_seconds()
    temp['time_diff_prev'] = temp['time_diff'].shift(-1)
    df['tweetstorm'] = temp.eval('time_diff < @time_threshold | \
                                 time_diff_prev < @time_threshold')
    return df
