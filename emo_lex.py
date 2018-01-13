import pandas as pd
import numpy as np
from src.load_pickle import load_pickle
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

(X_train, X_val, X_test,
 X_train_tfidf, X_val_tfidf, X_test_tfidf,
 X_train_pos, X_val_pos, X_test_pos,
 X_train_ner, X_val_ner, X_test_ner,
 y_train, y_val, y_test) = load_pickle('pickle/data.pkl')

filepath = ("NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], sep='\t')

emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()

emotions = emolex_words.columns.drop('word')

emo_df = pd.DataFrame(0, index=X_train.index, columns=emotions)

stemmer = SnowballStemmer("english")
for i, row in X_train[0:100].iterrows():
    document = word_tokenize(X_train.loc[i]['text'])
    for word in document:
        word = stemmer.stem(word.lower())
        emo_score = emolex_words[emolex_words.word == word]
        if not emo_score.empty:
            for emotion in (list(emo_score.columns)[1:]):
                emo_df.at[i, emotion] += emo_score[emotion]

X_train = pd.concat([X_train, emo_df], axis=1)
