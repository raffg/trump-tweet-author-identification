import pandas as pd
from src.load_data import load_json_list, apply_date_mask, sort_by_date


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


if __name__ == '__main__':
    main()
