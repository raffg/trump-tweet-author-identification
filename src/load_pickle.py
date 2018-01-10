import pickle


def main():
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     y_train, y_val, y_test) = load_pickle()


def load_pickle(file):
    pkl_file = open(file, 'rb')

    X_train = pickle.load(pkl_file)
    X_val = pickle.load(pkl_file)
    X_test = pickle.load(pkl_file)

    X_train_tfidf = pickle.load(pkl_file)
    X_val_tfidf = pickle.load(pkl_file)
    X_test_tfidf = pickle.load(pkl_file)

    X_train_pos = pickle.load(pkl_file)
    X_val_pos = pickle.load(pkl_file)
    X_test_pos = pickle.load(pkl_file)

    y_train = pickle.load(pkl_file)
    y_val = pickle.load(pkl_file)
    y_test = pickle.load(pkl_file)

    pkl_file.close()

    return (X_train, X_val, X_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            X_train_pos, X_val_pos, X_test_pos,
            y_train, y_val, y_test)


def load_pickle_ner(file):
    pkl_file = open(file, 'rb')

    X_train = pickle.load(pkl_file)
    X_val = pickle.load(pkl_file)
    X_test = pickle.load(pkl_file)

    X_train_tfidf = pickle.load(pkl_file)
    X_val_tfidf = pickle.load(pkl_file)
    X_test_tfidf = pickle.load(pkl_file)

    X_train_pos = pickle.load(pkl_file)
    X_val_pos = pickle.load(pkl_file)
    X_test_pos = pickle.load(pkl_file)

    X_train_ner = pickle.load(pkl_file)
    X_val_ner = pickle.load(pkl_file)
    X_test_ner = pickle.load(pkl_file)

    X_train_ner_tweetokenized = pickle.load(pkl_file)
    X_val_ner_tweetokenized = pickle.load(pkl_file)
    X_test_ner_tweetokenized = pickle.load(pkl_file)

    y_train = pickle.load(pkl_file)
    y_val = pickle.load(pkl_file)
    y_test = pickle.load(pkl_file)

    pkl_file.close()

    return (X_train, X_val, X_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            X_train_pos, X_val_pos, X_test_pos,
            X_train_ner, X_val_ner, X_test_ner,
            X_train_ner_tweetokenized,
            X_val_ner_tweetokenized,
            X_test_ner_tweetokenized,
            y_train, y_val, y_test)


if __name__ == '__main__':
    main()
