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


def period_of_day(df, timestamp):
    '''
    Takes a DataFrame and a specified column containing a timestamp and creates
    a new column indicating the period of the day in 6-hour increments
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with one new column
    '''

    new_df = df.copy()
    new_df['18-24'] = np.where(((new_df['created_at'].dt.hour >= 18) &
                               (new_df['created_at'].dt.hour < 24)),
                               True, False)
    new_df['12-17'] = np.where(((new_df['created_at'].dt.hour >= 12) &
                               (new_df['created_at'].dt.hour < 18)),
                               True, False)
    new_df['6-11'] = np.where(((new_df['created_at'].dt.hour >= 6) &
                               (new_df['created_at'].dt.hour < 12)),
                              True, False)
    new_df['0-5'] = np.where(new_df['created_at'].dt.hour < 6, True, False)
    return new_df
