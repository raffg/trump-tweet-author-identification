import pandas as pd
import numpy as np
import operator
from src.load_pickle import load_pickle
from ridge_feature_selection import plot_accuracies
from sklearn.naive_bayes import MultinomialNB


def main():
    run_model_naive_bayes('pickle/data_large.pkl')
    # naive_bayes_grid_search('pickle/data_large.pkl')


def naive_bayes_grid_search(file):
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_pickle(file)

    whole_train = pd.concat([X_train, X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    whole_val = pd.concat([X_val, X_val_pos,
                           X_val_tfidf, X_val_ner], axis=1)

    feat = np.load('all_train_features.npz')['arr_0']

    accuracies = []
    alphas = []

    for n in range(2, 20):
        for alpha in np.arange(2000, 3000, 10):
            accuracy = naive_bayes(np.array(whole_train[feat[:n]]),
                                   np.array(whole_val[feat[:n]]),
                                   np.array(y_train).ravel(),
                                   np.array(y_val).ravel(),
                                   alpha=alpha)
            accuracies.append(accuracy[1])
            alphas.append(alpha)
            print(n, alpha, accuracy[1])

    plot_accuracies(accuracies, "Naive Bayes")

    (max_index, max_value) = (max(enumerate(accuracies),
                              key=operator.itemgetter(1)))
    print(max_value, max_index)


def run_model_naive_bayes(file):
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

    naive_bayes_all_features = naive_bayes(np.array(X_train[feat]),
                                           np.array(X_val[feat]),
                                           np.array(y_train).ravel(),
                                           np.array(y_val).ravel())
    print('all features accuracy: ', naive_bayes_all_features)

    naive_bayes_text_accuracy = naive_bayes(np.array(X_train_tfidf),
                                            np.array(X_val_tfidf),
                                            np.array(y_train).ravel(),
                                            np.array(y_val).ravel())
    print('text accuracy: ', naive_bayes_text_accuracy)

    naive_bayes_pos = naive_bayes(np.array(X_train_pos),
                                  np.array(X_val_pos),
                                  np.array(y_train).ravel(),
                                  np.array(y_val).ravel())
    print('pos accuracy: ', naive_bayes_pos)

    naive_bayes_ner = naive_bayes(np.array(X_train_ner),
                                  np.array(X_val_ner),
                                  np.array(y_train).ravel(),
                                  np.array(y_val).ravel())
    print('ner accuracy: ', naive_bayes_ner)

    feat_text_train = pd.concat([X_train[feat], X_train_tfidf], axis=1)
    feat_text_val = pd.concat([X_val[feat], X_val_tfidf], axis=1)

    naive_bayes_all_features_text = naive_bayes(np.array(feat_text_train),
                                                np.array(feat_text_val),
                                                np.array(y_train).ravel(),
                                                np.array(y_val).ravel())
    print('all features with text tf-idf accuracy: ',
          naive_bayes_all_features_text)

    feat_pos_train = pd.concat([X_train[feat], X_train_pos], axis=1)
    feat_pos_val = pd.concat([X_val[feat], X_val_pos], axis=1)
    naive_bayes_all_features_pos = naive_bayes(np.array(feat_pos_train),
                                               np.array(feat_pos_val),
                                               np.array(y_train).ravel(),
                                               np.array(y_val).ravel())
    print('all features with pos tf-idf accuracy: ',
          naive_bayes_all_features_pos)

    feat_ner_train = pd.concat([X_train[feat], X_train_ner], axis=1)
    feat_ner_val = pd.concat([X_val[feat], X_val_ner], axis=1)
    naive_bayes_all_features_ner = naive_bayes(np.array(feat_ner_train),
                                               np.array(feat_ner_val),
                                               np.array(y_train).ravel(),
                                               np.array(y_val).ravel())
    print('all features with ner tf-idf accuracy: ',
          naive_bayes_all_features_ner)

    whole_train = pd.concat([X_train, X_train_pos,
                             X_train_tfidf, X_train_ner], axis=1)
    whole_val = pd.concat([X_val, X_val_pos,
                           X_val_tfidf, X_val_ner], axis=1)
    naive_bayes_whole = naive_bayes(np.array(whole_train[feat]),
                                    np.array(whole_val[feat]),
                                    np.array(y_train).ravel(),
                                    np.array(y_val).ravel())
    print('whole model accuracy: ', naive_bayes_whole)

    top_feat = np.load('all_train_features.npz')['arr_0'][:14]
    condensed_train = whole_train[top_feat]
    condensed_val = whole_val[top_feat]
    naive_bayes_condensed = naive_bayes(np.array(condensed_train),
                                        np.array(condensed_val),
                                        np.array(y_train).ravel(),
                                        np.array(y_val).ravel())
    print('condensed model accuracy: ', naive_bayes_condensed)


def naive_bayes(X_train, X_val, y_train, y_val, alpha=2500):
    # Basic Naive Bayes
    clf = MultinomialNB(alpha).fit(X_train, y_train)
    predicted = clf.predict(X_val)
    accuracy_train = np.mean(clf.predict(X_train) == y_train)
    accuracy_test = np.mean(predicted == y_val)
    return accuracy_train, accuracy_test


if __name__ == '__main__':
    main()
