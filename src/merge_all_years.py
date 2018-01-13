import pandas as pd
import pickle
from src.load_pickle import load_pickle
from src.save_pickle import tf_idf_matrix


def main():
    merge_all_years()


def merge_all_years():
    years = range(2009, 2018)

    X_train = pd.DataFrame()
    X_val = pd.DataFrame()
    X_test = pd.DataFrame()

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

        print('concatentating y_train')
        y_train = pd.concat([y_train, y_train1], axis=0)
        y_val = pd.concat([y_val, y_val1], axis=0)
        y_test = pd.concat([y_test, y_test1], axis=0)

        print('============================================================')
        print()

    # Create TF-IDF for text column
    print()
    print('TF-IDF on text column')
    tfidf_text = TfidfVectorizer(ngram_range=(1, 2),
                                 lowercase=False, token_pattern='\w+|\@\w+',
                                 norm='l2').fit(X_train['text'])
    cols = tfidf_text.get_feature_names()

    X_train_tfidf = tf_idf_matrix(X_train, 'text', tfidf_text, cols)
    X_val_tfidf = tf_idf_matrix(X_val, 'text', tfidf_text, cols)
    X_test_tfidf = tf_idf_matrix(X_test, 'text', tfidf_text, cols)

    # Create TF-IDF for pos column
    print()
    print('TF-IDF on pos column')
    tfidf_pos = TfidfVectorizer(ngram_range=(2, 3),
                                lowercase=False,
                                norm='l2').fit(X_train['pos'])
    cols = tfidf_pos.get_feature_names()

    X_train_pos = tf_idf_matrix(X_train, 'pos', tfidf_pos, cols)
    X_val_pos = tf_idf_matrix(X_val, 'pos', tfidf_pos, cols)
    X_test_pos = tf_idf_matrix(X_test, 'pos', tfidf_pos, cols)

    # Create TF-IDF for NER column
    print()
    print('TF-IDF on ner column')
    tfidf_ner = TfidfVectorizer(ngram_range=(1, 2),
                                lowercase=False,
                                norm='l2').fit(X_train['ner'])
    cols = tfidf_ner.get_feature_names()

    X_train_ner = tf_idf_matrix(X_train, 'ner', tfidf_ner, cols)
    X_val_ner = tf_idf_matrix(X_val, 'ner', tfidf_ner, cols)
    X_test_ner = tf_idf_matrix(X_test, 'ner', tfidf_ner, cols)

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
