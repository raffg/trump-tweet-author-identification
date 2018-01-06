import pandas as pd
from src.load_data import load_json_list, apply_date_mask, sort_by_date
from src.vader_sentiment import get_vader_scores, apply_vader
from src.style import apply_avg_lengths, mention_hashtag_url_columns


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

    # Create columns for average sentence and word length of tweet
    df = apply_avg_lengths(df, 'text')

    # Create columns for counts of @mentions, #hashtags, and urls
    df = mention_hashtag_url_columns(df, 'text')

    print(df[0:10])


if __name__ == '__main__':
    main()
