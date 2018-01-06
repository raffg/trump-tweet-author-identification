import re
import pandas as pd


def sentence_word_length(text):
    '''
    finds the average length of sentences and words in a given text
    INPUT: string
    OUTPUT: float(average sentence length), float(average word length)
    '''

    sentence_lengths = []
    word_lengths = []
    sentences = [s.strip() for s in re.split('[\.\?!]', text) if s]
    for sentence in sentences:
        words = sentence.split()
        word_lengths = word_lengths + [len(word) for word in words]
        sentence_length = len(words)
        sentence_lengths.append(sentence_length)
    return (sum(sentence_lengths) / float(len(sentence_lengths)),
            sum(word_lengths) / float(len(word_lengths)))


def tweet_length_column(df, column):
    '''
    takes a DataFrame and the name of a column of text and creates a new
    column containing the count of characters of the text
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame, with one new column
    '''

    df['tweet_length'] = df[column].str.len()
    return df


def apply_avg_lengths(df, column):
    '''
    takes a DataFrame with a specified column of text and adds two new columns
    to the DataFrame, corresponding to the average sentence and word lengths
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with two additional columns
    '''

    avg_lengths = pd.DataFrame(df[column].apply(sentence_word_length))
    unpacked = pd.DataFrame([d for idx, d in avg_lengths[column].iteritems()],
                            index=avg_lengths.index)
    unpacked.columns = ['avg_sentence_length', 'avg_word_length']
    return pd.concat([df, unpacked], axis=1)


def count_character(text, character):
    '''
    takes a text string and a character and outputs the number of occurances
    of that character in the text
    INPUT: text string, character string
    OUTPUT: int
    '''

    return text.count(character)


def punctuation_columns(df, column, punctuation_dict):
    '''
    takes a DataFrame, a column of text, and a dictionary with keys = character
    names and values = character, for example {'mentions':'@'}. Creates new
    columns containing the number of occurances of @mentions, #hashtags, urls,
    and specified punctuation
    INPUT: DataFrame, string of column name, dictionary
    OUTPUT: original DataFrame with three new columns
    '''

    for idx in range(len(punctuation_dict)):
        col = pd.DataFrame(df[column].apply(count_character,
                           character=list(punctuation_dict.values())[idx]))
        col.columns = [list(punctuation_dict.keys())[idx]]
        df = pd.concat([df, col], axis=1)

    return df
