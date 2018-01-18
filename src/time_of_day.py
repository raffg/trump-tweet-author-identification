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
    new_df['hour_20-02'] = np.where(((new_df['created_at'].dt.hour >= 20) |
                                    (new_df['created_at'].dt.hour < 2)),
                                    True, False)
    new_df['hour_14-20'] = np.where(((new_df['created_at'].dt.hour >= 14) &
                                    (new_df['created_at'].dt.hour < 20)),
                                    True, False)
    new_df['hour_08-14'] = np.where(((new_df['created_at'].dt.hour >= 8) &
                                    (new_df['created_at'].dt.hour < 14)),
                                    True, False)
    new_df['hour_02-08'] = np.where(((new_df['created_at'].dt.hour >= 2) &
                                    (new_df['created_at'].dt.hour < 8)),
                                    True, False)
    return new_df
