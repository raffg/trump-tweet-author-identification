import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize(feature_list, train, validation, test=False):
    '''
    Takes a list of numerical features to standardize, two DataFrames and an
    optional third DataFrame, and standardizes the feature columns. Outputs
    two or three DataFrames containing standardized feature columns and
    tf-idf columns.
    INPUT: list, DataFrame, DataFrame, optional DataFrame
    OUTPUT: two or three DataFrames
    '''

    scaler = StandardScaler()
    scaler.fit(train[feature_list])
    cols = train[feature_list].columns

    feat = feature_list.copy()
    feat.extend(['created_at', 'source', 'text',
                 'tweetokenize', 'pos', 'ner'])

    train_non_feature = train.drop(feat, axis=1)
    val_non_feature = validation.drop(feat, axis=1)

    train_data = pd.DataFrame(scaler.transform(train[feature_list]),
                              index=train.index, columns=cols)
    val_data = pd.DataFrame(scaler.transform(validation[feature_list]),
                            index=validation.index, columns=cols)

    train_data = pd.concat([train_data, train_non_feature], axis=1)
    val_data = pd.concat([val_data, val_non_feature], axis=1)

    if test is not False:
        test_non_feature = test.drop(feat, axis=1)
        test_data = pd.DataFrame(scaler.transform(test[feature_list]),
                                 index=test.index, columns=cols)
        test_data = pd.concat([test_data, test_non_feature], axis=1)
        return (train_data, val_data, test_data)

    return (train_data, val_data)
