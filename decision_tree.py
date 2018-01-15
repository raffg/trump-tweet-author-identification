import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from sklearn.tree import DecisionTreeClassifier


def main():
    run_model_decision_tree('pickle/data_large.pkl')


def run_model_decision_tree(file):
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

    decision_tree_all_features = decision_tree(np.array(X_train[feat]),
                                               np.array(X_val[feat]),
                                               np.array(y_train).ravel(),
                                               np.array(y_val).ravel())
    print('all features accuracy: ', decision_tree_all_features)

    decision_tree_text_accuracy = decision_tree(np.array(X_train_tfidf),
                                                np.array(X_val_tfidf),
                                                np.array(y_train).ravel(),
                                                np.array(y_val).ravel())
    print('text accuracy: ', decision_tree_text_accuracy)

    decision_tree_pos = decision_tree(np.array(X_train_pos),
                                      np.array(X_val_pos),
                                      np.array(y_train).ravel(),
                                      np.array(y_val).ravel())
    print('pos accuracy: ', decision_tree_pos)

    decision_tree_ner = decision_tree(np.array(X_train_ner),
                                      np.array(X_val_ner),
                                      np.array(y_train).ravel(),
                                      np.array(y_val).ravel())
    print('ner accuracy: ', decision_tree_ner)

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    feat_text_val = pd.concat([X_val[feat], X_val_tfidf], axis=1)
    decision_tree_all_features_text = decision_tree(np.array(feat_text_train),
                                                    np.array(feat_text_val),
                                                    np.array(y_train).ravel(),
                                                    np.array(y_val).ravel())
    print('all features with text tf-idf accuracy: ',
          decision_tree_all_features_text)

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val[feat], X_val_pos], axis=1)
    decision_tree_all_features_pos = decision_tree(np.array(feat_pos_train),
                                                   np.array(feat_pos_val),
                                                   np.array(y_train).ravel(),
                                                   np.array(y_val).ravel())
    print('all features with pos tf-idf accuracy: ',
          decision_tree_all_features_pos)

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    feat_ner_val = pd.concat([X_val[feat], X_val_ner], axis=1)
    decision_tree_all_features_ner = decision_tree(np.array(feat_ner_train),
                                                   np.array(feat_ner_val),
                                                   np.array(y_train).ravel(),
                                                   np.array(y_val).ravel())
    print('all features with ner tf-idf accuracy: ',
          decision_tree_all_features_ner)

    feat = np.load('all_train_features.npz')['arr_0'][:18]

    whole_train = pd.concat([X_train, X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    whole_val = pd.concat([X_val, X_val_pos,
                           X_val_tfidf, X_val_ner], axis=1)
    decision_tree_whole = decision_tree(np.array(whole_train[feat]),
                                        np.array(whole_val[feat]),
                                        np.array(y_train).ravel(),
                                        np.array(y_val).ravel())
    print('whole model accuracy: ', decision_tree_whole)

    top_feat = np.load('top_features.npz')['arr_0']
    condensed_train = whole_train[top_feat]
    condensed_val = whole_val[top_feat]
    decision_tree_condensed = decision_tree(np.array(condensed_train),
                                            np.array(condensed_val),
                                            np.array(y_train).ravel(),
                                            np.array(y_val).ravel())
    print('condensed model accuracy: ', decision_tree_condensed)


def decision_tree(X_train, X_val, y_train, y_val):
    # Basic decision tree
    dt = DecisionTreeClassifier('max_depth': 20,
                                'max_features': 'sqrt',
                                'max_leaf_nodes': 100,
                                'min_samples_leaf': 2,
                                'min_samples_split': 2).fit(X_train, y_train)
    predicted = dt.predict(X_val)
    accuracy_train = np.mean(dt.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)
    return accuracy_train, accuracy_test


if __name__ == '__main__':
    main()
