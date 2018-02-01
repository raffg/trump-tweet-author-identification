import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    run_model_gb()


def run_model_gb():
    X_train = pd.read_pickle('pickle/train_all_std.pkl')
    X_val = pd.read_pickle('pickle/test_all_std.pkl')
    y_train = pd.read_pickle('pickle/y_train_all_std.pkl')
    y_val = pd.read_pickle('pickle/y_test_all_std.pkl')

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

    print('all features')
    gb_all_features = gb(np.array(X_train[feat]),
                         np.array(X_val[feat]),
                         np.array(y_train).ravel(),
                         np.array(y_val).ravel())

    whole_train = X_train.drop(drop, axis=1)
    whole_val = X_val.drop(drop, axis=1)
    '''
    print('whole model')
    random_forest_whole = random_forest(np.array(whole_train),
                                        np.array(whole_val),
                                        np.array(y_train).ravel(),
                                        np.array(y_val).ravel())
    # random_forest_save_pickle(random_forest_whole)
    '''
    top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:300])
    train_feat = []
    val_feat = []
    for feat in top_feat:
        if feat in whole_train.columns:
            train_feat.append(feat)
        if feat in whole_val.columns:
            val_feat.append(feat)

    print('condensed model')
    condensed_train = whole_train[train_feat]
    condensed_val = whole_val[val_feat]
    gb_condensed = gb(np.array(condensed_train),
                      np.array(condensed_val),
                      np.array(y_train).ravel(),
                      np.array(y_val).ravel())

    feats = list(zip(top_feat, gb_condensed.feature_importances_))
    feats = sorted(feats, key=lambda x: x[1])
    feats = [x[0] for x in feats][::-1]
    print(feats)

    # gb_save_pickle(gb_condensed)


def gb(X_train, X_val, y_train, y_val):
    # Basic Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200,
                                    learning_rate=.1,
                                    max_depth=6,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    subsample=1,
                                    max_features=None
                                    ).fit(X_train, y_train)
    predicted = gb.predict(X_val)

    print('Accuracy: ', accuracy_score(y_val, predicted))
    print('Precision: ', precision_score(y_val, predicted))
    print('Recall: ', recall_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted))
    print()

    return gb


def gb_save_pickle(model):
    # Save pickle file
    output = open('pickle/gradient_boosting_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
