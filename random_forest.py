import pandas as pd
import numpy as np
import pickle
from src.load_pickle import load_pickle
from src.standardize import standardize
from src.cross_val_data import cross_val_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    run_model_random_forest()
    # random_forest_save_pickle()


def run_model_random_forest():
    X_train = pd.read_pickle('pickle/train_val_all.pkl')
    X_val = pd.read_pickle('pickle/val_all.pkl')
    y_train = pd.read_pickle('pickle/y_train_val_all.pkl')
    y_val = pd.read_pickle('pickle/y_val_all.pkl')

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    drop = ['created_at', 'id_str', 'in_reply_to_user_id_str', 'tweetokenize',
            'text', 'pos', 'ner']

    random_forest_all_features = random_forest(np.array(X_train[feat]),
                                               np.array(X_val[feat]),
                                               np.array(y_train).ravel(),
                                               np.array(y_val).ravel())
    print('all features accuracy: ', random_forest_all_features)

    whole_train = X_train.drop(drop, axis=1)
    whole_val = X_val.drop(drop, axis=1)

    random_forest_whole = random_forest(np.array(whole_train),
                                        np.array(whole_val),
                                        np.array(y_train).ravel(),
                                        np.array(y_val).ravel())
    print('whole model accuracy: ', random_forest_whole)

    # top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:100])
    # train_feat = []
    # val_feat = []
    # for feat in top_feat:
    #     if feat in whole_train.columns:
    #         train_feat.append(feat)
    #     if feat in whole_val.columns:
    #         val_feat.append(feat)
    #
    # condensed_train = whole_train[train_feat]
    # condensed_val = whole_val[val_feat]
    # random_forest_condensed = random_forest(np.array(condensed_train),
    #                                         np.array(condensed_val),
    #                                         np.array(y_train).ravel(),
    #                                         np.array(y_val).ravel())
    # print('condensed model accuracy: ', random_forest_condensed)


def random_forest(X_train, X_val, y_train, y_val):
    # Basic random forest
    rf = RandomForestClassifier(max_depth=20,
                                max_features='sqrt',
                                max_leaf_nodes=None,
                                min_samples_leaf=2,
                                min_samples_split=2,
                                n_estimators=1000,
                                n_jobs=-1).fit(X_train, y_train)
    predicted = rf.predict(X_val)
    accuracy_train = np.mean(rf.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)

    print()
    print('Accuracy: ', accuracy_score(y_val, predicted))
    print('Precision: ', precision_score(y_val, predicted))
    print('Recall: ', recall_score(y_val, predicted))

    # Save pickle file
    # output = open('pickle/random_forest_model', 'wb')
    # print('Pickle dump model')
    # pickle.dump(rf, output, protocol=4)
    # output.close()

    return accuracy_train, accuracy_test


def random_forest_save_pickle():
    # Basic random forest, save pickle

    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle('pickle/data.pkl')

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    (X_train, X_train_tfidf, X_train_pos, X_train_ner,
     X_test, X_test_tfidf, X_test_pos, X_test_ner) = cross_val_data(X_train,
                                                                    X_val,
                                                                    X_test)
    (X_train, X_test) = standardize(feat, X_train, X_test)

    # Concatenate all training DataFrames
    X_train = pd.concat([X_train, X_train_tfidf,
                         X_train_pos, X_train_ner], axis=1)
    X_test = pd.concat([X_test, X_test_tfidf,
                        X_test_pos, X_test_ner], axis=1)
    y_train = pd.concat([y_train, y_val], axis=0)
    y_train = np.array(y_train).ravel()

    rf = RandomForestClassifier(max_depth=20,
                                max_features='sqrt',
                                max_leaf_nodes=None,
                                min_samples_leaf=2,
                                min_samples_split=2,
                                n_estimators=1000,
                                n_jobs=-1).fit(X_train, y_train)

    # Save pickle file
    output = open('pickle/random_forest_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(rf, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
