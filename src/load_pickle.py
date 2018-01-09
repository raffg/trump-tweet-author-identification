import pickle

def load_pickle():
    pkl_file = open('data.pkl', 'rb')

    X_train = pickle.load(pkl_file)
    X_val = pickle.load(pkl_file)
    X_test = pickle.load(pkl_file)
    y_train = pickle.load(pkl_file)
    y_val = pickle.load(pkl_file)
    y_test = pickle.load(pkl_file)
    feat_train = pickle.load(pkl_file)
    feat_val = pickle.load(pkl_file)
    feat_test = pickle.load(pkl_file)

    pkl_file.close()

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            feat_train, feat_val, feat_test)
