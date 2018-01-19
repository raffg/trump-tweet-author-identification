import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from sklearn.ensemble import RandomForestClassifier


def main():
    run_model_random_forest('pickle/data.pkl')


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
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    random_forest_all_features = random_forest(np.array(X_train[feat]),
                                               np.array(X_val[feat]),
                                               np.array(y_train).ravel(),
                                               np.array(y_val).ravel())
    print('all features accuracy: ', random_forest_all_features)

    random_forest_text_accuracy = random_forest(np.array(X_train_tfidf),
                                                np.array(X_val_tfidf),
                                                np.array(y_train).ravel(),
                                                np.array(y_val).ravel())
    print('text accuracy: ', random_forest_text_accuracy)

    random_forest_pos = random_forest(np.array(X_train_pos),
                                      np.array(X_val_pos),
                                      np.array(y_train).ravel(),
                                      np.array(y_val).ravel())
    print('pos accuracy: ', random_forest_pos)

    random_forest_ner = random_forest(np.array(X_train_ner),
                                      np.array(X_val_ner),
                                      np.array(y_train).ravel(),
                                      np.array(y_val).ravel())
    print('ner accuracy: ', random_forest_ner)

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    feat_text_val = pd.concat([X_val[feat], X_val_tfidf], axis=1)
    random_forest_all_features_text = random_forest(np.array(feat_text_train),
                                                    np.array(feat_text_val),
                                                    np.array(y_train).ravel(),
                                                    np.array(y_val).ravel())
    print('all features with text tf-idf accuracy: ',
          random_forest_all_features_text)

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val[feat], X_val_pos], axis=1)
    random_forest_all_features_pos = random_forest(np.array(feat_pos_train),
                                                   np.array(feat_pos_val),
                                                   np.array(y_train).ravel(),
                                                   np.array(y_val).ravel())
    print('all features with pos tf-idf accuracy: ',
          random_forest_all_features_pos)

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    feat_ner_val = pd.concat([X_val[feat], X_val_ner], axis=1)
    random_forest_all_features_ner = random_forest(np.array(feat_ner_train),
                                                   np.array(feat_ner_val),
                                                   np.array(y_train).ravel(),
                                                   np.array(y_val).ravel())
    print('all features with ner tf-idf accuracy: ',
          random_forest_all_features_ner)

    whole_train = pd.concat([X_train, X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    whole_val = pd.concat([X_val, X_val_pos,
                           X_val_tfidf, X_val_ner], axis=1)

    random_forest_whole = random_forest(np.array(whole_train[feat]),
                                        np.array(whole_val[feat]),
                                        np.array(y_train).ravel(),
                                        np.array(y_val).ravel())
    print('whole model accuracy: ', random_forest_whole)

    top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:100])
    train_feat = []
    val_feat = []
    for feat in top_feat:
        if feat in whole_train.columns:
            train_feat.append(feat)
        if feat in whole_val.columns:
            val_feat.append(feat)

    condensed_train = whole_train[train_feat]
    condensed_val = whole_val[val_feat]
    random_forest_condensed = random_forest(np.array(condensed_train),
                                            np.array(condensed_val),
                                            np.array(y_train).ravel(),
                                            np.array(y_val).ravel())
    print('condensed model accuracy: ', random_forest_condensed)


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
    return accuracy_train, accuracy_test


if __name__ == '__main__':
    main()
