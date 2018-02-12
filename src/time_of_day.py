import pandas as pd
import numpy as np


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


def period_of_day(df, timestamp):
    '''
    Takes a DataFrame and a specified column containing a timestamp and creates
    a new column indicating the period of the day in 6-hour increments
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with one new column
    '''

    new_df = df.copy()
    new_df['hour_20_02'] = np.where(((new_df['created_at'].dt.hour >= 20) |
                                    (new_df['created_at'].dt.hour < 2)),
                                    True, False)
    new_df['hour_14_20'] = np.where(((new_df['created_at'].dt.hour >= 14) &
                                    (new_df['created_at'].dt.hour < 20)),
                                    True, False)
    new_df['hour_08_14'] = np.where(((new_df['created_at'].dt.hour >= 8) &
                                    (new_df['created_at'].dt.hour < 14)),
                                    True, False)
    new_df['hour_02_08'] = np.where(((new_df['created_at'].dt.hour >= 2) &
                                    (new_df['created_at'].dt.hour < 8)),
                                    True, False)
    return new_df


def day_of_week(df, timestamp):
    '''
    Takes a DataFrame and a specified column containing a timestamp and creates
    a new column indicating the day of the week
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with one new column
    '''
    new_df = df.copy()
    new_df['day_of_week'] = new_df[timestamp].dt.weekday

    return new_df


def weekend(df, day_of_week):
    '''
    Takes a DataFrame and a specified column containing a day of the week and
    creates a new column indicating if the day occurs on a weekend
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with one new column
    '''
    new_df = df.copy()
    new_df['weekend'] = new_df[day_of_week].apply(lambda x: 1 if x in [5, 6] else 0)

    return new_df
