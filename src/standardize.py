import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize(feature_list, train, validation, test=False):
    '''
    Takes a list of numerical features to standardize, two DataFrames and an
    optional third DataFrame, and standardizes the feature columns. Outputs
    two or three DataFrames containing only the feature columns.
    INPUT: list, DataFrame, DataFrame, optional DataFrame
    OUTPUT: two or three DataFrames
    '''

    scaler = StandardScaler()
    scaler.fit(train[feature_list])
    cols = train[feature_list].columns

    train_data = pd.DataFrame(scaler.transform(train[feature_list]),
                              index=train.index, columns=cols)
    val_data = pd.DataFrame(scaler.transform(validation[feature_list]),
                            index=validation.index, columns=cols)
    if test is not False:
        test_data = pd.DataFrame(scaler.transform(test[feature_list]),
                                 index=test.index, columns=cols)
        return (train_data, val_data, test_data)

    return (train_data, val_data)
