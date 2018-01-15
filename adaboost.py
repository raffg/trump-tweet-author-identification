import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from sklearn.ensemble import AdaBoostClassifier


def main():
    run_model_random_forest('pickle/data_large.pkl')


def run_model_random_forest(file):
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
            'all_caps', 'tweetstorm', 'hour', 'period_1', 'period_2',
            'period_3', 'period_4']

    adaboost_all_features = adaboost(np.array(X_train[feat]),
                                     np.array(X_val[feat]),
                                     np.array(y_train).ravel(),
                                     np.array(y_val).ravel())
    print('all features accuracy: ', adaboost_all_features)

    adaboost_text_accuracy = adaboost(np.array(X_train_tfidf),
                                      np.array(X_val_tfidf),
                                      np.array(y_train).ravel(),
                                      np.array(y_val).ravel())
    print('text accuracy: ', adaboost_text_accuracy)

    adaboost_pos = adaboost(np.array(X_train_pos),
                            np.array(X_val_pos),
                            np.array(y_train).ravel(),
                            np.array(y_val).ravel())
    print('pos accuracy: ', adaboost_pos)

    adaboost_ner = adaboost(np.array(X_train_ner),
                            np.array(X_val_ner),
                            np.array(y_train).ravel(),
                            np.array(y_val).ravel())
    print('ner accuracy: ', adaboost_ner)

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    feat_text_val = pd.concat([X_val[feat], X_val_tfidf], axis=1)
    adaboost_all_features_text = adaboost(np.array(feat_text_train),
                                          np.array(feat_text_val),
                                          np.array(y_train).ravel(),
                                          np.array(y_val).ravel())
    print('all features with text tf-idf accuracy: ',
          adaboost_all_features_text)

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val[feat], X_val_pos], axis=1)
    adaboost_all_features_pos = adaboost(np.array(feat_pos_train),
                                         np.array(feat_pos_val),
                                         np.array(y_train).ravel(),
                                         np.array(y_val).ravel())
    print('all features with pos tf-idf accuracy: ',
          adaboost_all_features_pos)

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    feat_ner_val = pd.concat([X_val[feat], X_val_ner], axis=1)
    adaboost_all_features_ner = adaboost(np.array(feat_ner_train),
                                         np.array(feat_ner_val),
                                         np.array(y_train).ravel(),
                                         np.array(y_val).ravel())
    print('all features with ner tf-idf accuracy: ',
          adaboost_all_features_ner)

    feat = np.load('all_train_features.npz')['arr_0'][:18]

    whole_train = pd.concat([X_train, X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    whole_val = pd.concat([X_val, X_val_pos,
                           X_val_tfidf, X_val_ner], axis=1)
    adaboost_whole = adaboost(np.array(whole_train[feat]),
                              np.array(whole_val[feat]),
                              np.array(y_train).ravel(),
                              np.array(y_val).ravel())
    print('whole model accuracy: ', adaboost_whole)

    top_feat = np.load('top_features.npz')['arr_0']
    condensed_train = whole_train[top_feat]
    condensed_val = whole_val[top_feat]
    adaboost_condensed = adaboost(np.array(condensed_train),
                                  np.array(condensed_val),
                                  np.array(y_train).ravel(),
                                  np.array(y_val).ravel())
    print('condensed model accuracy: ', adaboost_condensed)


def adaboost(X_train, X_val, y_train, y_val):
    # Basic AdaBoost classifier
    ab = AdaBoostClassifier().fit(X_train, y_train)
    predicted = ab.predict(X_val)
    accuracy_train = np.mean(ab.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)
    return accuracy_train, accuracy_test


if __name__ == '__main__':
    main()
