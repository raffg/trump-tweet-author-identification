import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from src.standardize import standardize
from sklearn.neighbors import KNeighborsClassifier


def main():
    run_model_knn('pickle/data.pkl')


def run_model_knn(file):
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle(file)

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08']

    (X_train, X_val, X_test) = standardize(feat, X_train, X_val, X_test)

    knn_all_features = knn(np.array(X_train[feat]),
                           np.array(X_val[feat]),
                           np.array(y_train).ravel(),
                           np.array(y_val).ravel())
    print('all features accuracy: ', knn_all_features)

    knn_text_accuracy = knn(np.array(X_train_tfidf),
                            np.array(X_val_tfidf),
                            np.array(y_train).ravel(),
                            np.array(y_val).ravel())
    print('text accuracy: ', knn_text_accuracy)

    knn_pos = knn(np.array(X_train_pos),
                  np.array(X_val_pos),
                  np.array(y_train).ravel(),
                  np.array(y_val).ravel())
    print('pos accuracy: ', knn_pos)

    knn_ner = knn(np.array(X_train_ner),
                  np.array(X_val_ner),
                  np.array(y_train).ravel(),
                  np.array(y_val).ravel())
    print('ner accuracy: ', knn_ner)

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    feat_text_val = pd.concat([X_val[feat], X_val_tfidf], axis=1)

    knn_all_features_text = knn(np.array(feat_text_train),
                                np.array(feat_text_val),
                                np.array(y_train).ravel(),
                                np.array(y_val).ravel())
    print('all features with text tf-idf accuracy: ',
          knn_all_features_text)

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val[feat], X_val_pos], axis=1)
    knn_all_features_pos = knn(np.array(feat_pos_train),
                               np.array(feat_pos_val),
                               np.array(y_train).ravel(),
                               np.array(y_val).ravel())
    print('all features with pos tf-idf accuracy: ',
          knn_all_features_pos)

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    feat_ner_val = pd.concat([X_val[feat], X_val_ner], axis=1)
    knn_all_features_ner = knn(np.array(feat_ner_train),
                               np.array(feat_ner_val),
                               np.array(y_train).ravel(),
                               np.array(y_val).ravel())
    print('all features with ner tf-idf accuracy: ',
          knn_all_features_ner)

    whole_train = pd.concat([X_train[feat], X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    whole_val = pd.concat([X_val[feat], X_val_pos,
                           X_val_tfidf, X_val_ner], axis=1)
    knn_whole = knn(np.array(whole_train),
                    np.array(whole_val),
                    np.array(y_train).ravel(),
                    np.array(y_val).ravel())
    print('whole model accuracy: ', knn_whole)

    top_feat = np.load('pickle/top_features.npz')['arr_0'][:20]
    condensed_train = whole_train[top_feat]
    condensed_val = whole_val[top_feat]
    knn_condensed = knn(np.array(condensed_train),
                        np.array(condensed_val),
                        np.array(y_train).ravel(),
                        np.array(y_val).ravel())
    print('condensed model accuracy: ', knn_condensed)


def knn(X_train, X_val, y_train, y_val):
    # Basic K Nearest Neighbors Classifier
    clf = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    predicted = clf.predict(X_val)
    accuracy_train = np.mean(clf.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)
    return accuracy_train, accuracy_test


if __name__ == '__main__':
    main()
