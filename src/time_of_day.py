import pandas as pd


def time_of_day(df, timestamp):
    '''
    Takes a DataFrame and a specified column containing a timestamp and creates
    a new column indicating the hour of the day
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with one new column
    '''

    new_df = df.copy()
    new_df['hour'] = new_df[timestamp].dt.hour
    return new_df
