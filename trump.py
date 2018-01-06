import pandas as pd
from src.load_data import load_json_list, apply_date_mask, sort_by_date
from src.vader_sentiment import apply_vader
from src.style import apply_avg_lengths, tweet_length_column, \
                      punctuation_columns


def main():
    # Load and sort the data
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

    #==========================================================================
    # Testing
    df = df[0:10]
    #==========================================================================

    # Create columns for vader sentiment
    df = apply_vader(df, 'text')

    # Create columns for average tweet, sentence, and word length of tweet
    df = tweet_length_column(df, 'text')
    df = apply_avg_lengths(df, 'text')

    # Create columns for counts of @mentions, #hashtags, urls, and punctuation
    punctuation_dict = {'mentions': '@', 'hashtags': '#', 'urls': '://',
                        'commas': ',', 'semicolons': ';', 'exclamations': '!',
                        'periods': '.', 'questions': '?'}

    df = punctuation_columns(df, 'text', punctuation_dict)

    print(df)


if __name__ == '__main__':
    main()
