import pandas as pd
import pickle
from src.load_pickle import load_pickle


def main():
    merge_all_years()


def merge_all_years():
    years = range(2009, 2018)

    X_train = pd.DataFrame()
    X_val = pd.DataFrame()
    X_test = pd.DataFrame()
    X_train_tfidf = pd.DataFrame()
    X_val_tfidf = pd.DataFrame()
    X_test_tfidf = pd.DataFrame()
    X_train_pos = pd.DataFrame()
    X_val_pos = pd.DataFrame()
    X_test_pos = pd.DataFrame()
    X_train_ner = pd.DataFrame()
    X_val_ner = pd.DataFrame()
    X_test_ner = pd.DataFrame()
    y_train = pd.DataFrame()
    y_val = pd.DataFrame()
    y_test = pd.DataFrame()

    for year in years:
        print('-----Opening ' + str(year) + '-----')
        (X_train1, X_val1, X_test1,
         X_train_tfidf1, X_val_tfidf1, X_test_tfidf1,
         X_train_pos1, X_val_pos1, X_test_pos1,
         X_train_ner1, X_val_ner1, X_test_ner1,
         y_train1, y_val1, y_test1) = load_pickle('pickle/' +
                                                  str(year) +
                                                  '.pkl')

        print('concatentating X_train')
        X_train = pd.concat([X_train, X_train1], axis=0)
        X_val = pd.concat([X_val, X_val1], axis=0)
        X_test = pd.concat([X_test, X_test1], axis=0)

        print('concatentating X_train_tfidf')
        X_train_tfidf = pd.concat([X_train_tfidf, X_train_tfidf1], axis=0)
        X_val_tfidf = pd.concat([X_val_tfidf, X_val_tfidf1], axis=0)
        X_test_tfidf = pd.concat([X_test_tfidf, X_test_tfidf1], axis=0)

        print('concatentating X_train_pos')
        X_train_pos = pd.concat([X_train_pos, X_train_pos1], axis=0)
        X_val_pos = pd.concat([X_val_pos, X_val_pos1], axis=0)
        X_test_pos = pd.concat([X_test_pos, X_test_pos1], axis=0)

        print('concatentating X_train_ner')
        X_train_ner = pd.concat([X_train_ner, X_train_ner1], axis=0)
        X_val_ner = pd.concat([X_val_ner, X_val_ner1], axis=0)
        X_test_ner = pd.concat([X_test_ner, X_test_ner1], axis=0)

        print('concatentating y_train')
        y_train = pd.concat([y_train, y_train1], axis=0)
        y_val = pd.concat([y_val, y_val1], axis=0)
        y_test = pd.concat([y_test, y_test1], axis=0)

        print('============================================================')
        print()

    output = open('pickle/all_data.pkl', 'wb')
    print()

    print('Pickle dump X_train')
    pickle.dump(X_train, output, protocol=4)
    print('Pickle dump X_val')
    pickle.dump(X_val, output, protocol=4)
    print('Pickle dump X_test')
    pickle.dump(X_test, output, protocol=4)

    print('Pickle dump X_train_tfidf')
    pickle.dump(X_train_tfidf, output, protocol=4)
    print('Pickle dump X_val_tfidf')
    pickle.dump(X_val_tfidf, output, protocol=4)
    print('Pickle dump X_test_tfidf')
    pickle.dump(X_test_tfidf, output, protocol=4)

    print('Pickle dump X_train_pos')
    pickle.dump(X_train_pos, output, protocol=4)
    print('Pickle dump X_val_pos')
    pickle.dump(X_val_pos, output, protocol=4)
    print('Pickle dump X_test_pos')
    pickle.dump(X_test_pos, output, protocol=4)

    print('Pickle dump X_train_ner')
    pickle.dump(X_train_ner, output, protocol=4)
    print('Pickle dump X_val_ner')
    pickle.dump(X_val_ner, output, protocol=4)
    print('Pickle dump X_test_ner')
    pickle.dump(X_test_ner, output, protocol=4)

    print('Pickle dump y_train')
    pickle.dump(y_train, output, protocol=4)
    print('Pickle dump y_val')
    pickle.dump(y_val, output, protocol=4)
    print('Pickle dump y_test')
    pickle.dump(y_test, output, protocol=4)

    output.close()


if __name__ == '__main__':
    main()
