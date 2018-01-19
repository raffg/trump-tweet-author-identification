import pandas as pd
import numpy as np
from itertools import combinations


def cross_product_features(df, column_list):
    '''
    Takes a DataFrame and a list of feature columns, and creates new feature
    columns of the cross product of each combination of feature columns.
    Each new feature column takes it's name from the new_feature_list
    INPUT: DataFrame, list of column names, list of feature names
    OUTPUT: the original DataFrame with new feature columns
    '''

    new_df = df.copy()

    combo_list = list(combinations(column_list, 2))

    for combo in combo_list:
        feature_name = str(combo[0]) + '__' + str(combo[1])
        feature = (new_df[combo[0]] * 1.) * (new_df[combo[1]] * 1.)
        new_df[feature_name] = feature

    return new_df
