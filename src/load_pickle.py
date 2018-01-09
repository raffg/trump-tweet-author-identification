import pickle

def load_pickle():
    pkl_file = open('data.pkl', 'rb')

    X_train = pickle.load(pkl_file)
    X_val = pickle.load(pkl_file)
    X_test = pickle.load(pkl_file)

    X_train_tfidf = pickle.load(pkl_file)
    X_val_tfidf = pickle.load(pkl_file)
    X_test_tfidf = pickle.load(pkl_file)

    X_train_pos = pickle.load(pkl_file)
    X_val_pos = pickle.load(pkl_file)
    feat_X_test_postest = pickle.load(pkl_file)

    y_train = pickle.load(pkl_file)
    y_val = pickle.load(pkl_file)
    y_test = pickle.load(pkl_file)

    pkl_file.close()

    return (X_train, X_val, X_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            X_train_pos, X_val_pos, X_test_pos,
            y_train, y_val, y_test)
