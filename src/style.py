import re
import pandas as pd


def sentence_word_length(text):
    '''
    Finds the average length of sentences and words in a given text
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


def apply_avg_lengths(df, column):
    '''
    Takes a DataFrame with a specified column of text and adds two new columns
    to the DataFrame, corresponding to the average sentence and word lengths
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with two additional columns
    '''

    avg_lengths = pd.DataFrame(df[column].apply(sentence_word_length))
    unpacked = pd.DataFrame([d for idx, d in avg_lengths[column].iteritems()],
                            index=avg_lengths.index)
    unpacked.columns = ['avg_sentence_length', 'avg_word_length']
    return pd.concat([df, unpacked], axis=1)


def tweet_length(df, column):
    '''
    Takes a DataFrame and the name of a column of text and creates a new
    column containing the count of characters of the text
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame, with one new column
    '''

    new_df = df.copy()
    new_df['tweet_length'] = new_df[column].str.len()
    return new_df


def count_character(text, character):
    '''
    Takes a text string and a character and outputs the number of occurances
    of that character in the text
    INPUT: text string, character string
    OUTPUT: int
    '''

    return text.count(character)


def punctuation_columns(df, column, punctuation_dict):
    '''
    Takes a DataFrame, a column of text, and a dictionary with keys = character
    names and values = character, for example {'mentions':'@'}. Creates new
    columns containing the number of occurances of @mentions, #hashtags, urls,
    and specified punctuation
    INPUT: DataFrame, string of column name, dictionary
    OUTPUT: original DataFrame with three new columns
    '''

    new_df = df.copy()
    for idx in range(len(punctuation_dict)):
        col = pd.DataFrame(df[column].apply(count_character,
                           character=list(punctuation_dict.values())[idx]))
        col.columns = [list(punctuation_dict.keys())[idx]]
        new_df = pd.concat([df, col], axis=1)

    return new_df


def identify_quoted_retweet(text):
    '''
    Takes a string of text and returns 1 if the text begins with '"@' and a 0
    if not
    INPUT: string
    OUTPUT: int
    '''

    return (False if re.match('^"@', text) is None else True)


def quoted_retweet(df, column):
    '''
    Takes a DataFrame and a column of text and creates a new colun with 1 if
    the text is fully surrounded by quote marks and a 0 if not
    INPUT: DataFrame, String of column name
    OUPUT: original DataFrame with one new column
    '''

    quote = pd.DataFrame(df[column].apply(identify_quoted_retweet),
                         index=df.index)
    quote.columns = ['is_quoted_retweet']
    return pd.concat([df, quote], axis=1)
