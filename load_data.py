import pandas as pd


def main():
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
    masked_df = apply_date_mask(raw_data, 'created_at',
                                '2015-06-01', '2017-03-26')
    df = sort_by_date(masked_df, 'created_at')
    print (len(df))


def load_json_list(list):
    '''
    takes a list of json files, loads them, and concatenates them
    INPUT: a list of json files
    OUTPUT: a single concatenated DataFrame
    '''

    files = []
    for file in list:
        df = pd.read_json(file)
        files.append(df)
    return pd.concat(files)


def apply_date_mask(df, date_column, start_date, end_date):
    '''
    applies mask to a df to include only dates within the given date range
    INPUT: a DataFrame, the name of the datetime column, start and end dates
    OUTPUT: a DataFrame with a datetime index, sorted by datetime
    '''

    mask = (df[date_column] > start_date) & (df[date_column] <= end_date)
    return df.loc[mask]


def sort_by_date(df, date_column):
    '''
    takes a DataFrame with a column in datetime format and sorts by date_column
    INPUT: a DataFrame and the name of a column in datetime format
    OUTPUT: a DataFrame with a datetime index
    '''

    sorted_data = df.sort_values(date_column)
    data = sorted_data.set_index(date_column)
    return data


if __name__ == '__main__':
    main()
