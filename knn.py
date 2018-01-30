import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    run_model_knn()


def run_model_knn():
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

    whole_train = X_train.drop(drop, axis=1)
    whole_val = X_val.drop(drop, axis=1)

    top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:13])
    train_feat = []
    val_feat = []
    for feat in top_feat:
        if feat in whole_train.columns:
            train_feat.append(feat)
        if feat in whole_val.columns:
            val_feat.append(feat)

    pca = PCA(n_components=12)
    pca.fit(whole_train[train_feat])
    whole_train = pca.transform(whole_train[train_feat])
    whole_val = pca.transform(whole_val[val_feat])

    print('condensed model')
    condensed_train = whole_train
    condensed_val = whole_val
    knn_condensed = knn(np.array(condensed_train),
                        np.array(condensed_val),
                        np.array(y_train).ravel(),
                        np.array(y_val).ravel())
    # knn_save_pickle(knn_condensed)
    # knn_save_pca(pca)


def knn(X_train, X_val, y_train, y_val):
    # Basic knn
    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    predicted = knn.predict(X_val)
    accuracy_train = np.mean(knn.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)

    print('Accuracy: ', accuracy_score(y_val, predicted))
    print('Precision: ', precision_score(y_val, predicted))
    print('Recall: ', recall_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted))
    print()

    return knn


def knn_save_pickle(model):
    # Save pickle file
    output = open('pickle/knn_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


def knn_save_pca(pca):
    # Save pickle file
    output = open('pickle/knn_pca.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(pca, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
