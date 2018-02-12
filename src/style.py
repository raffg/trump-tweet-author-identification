import re
import pandas as pd
from src import tweetokenizer as t


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
    names and values = character, for example {'comma':','}. Creates new
    columns containing the number of occurances specified punctuation
    INPUT: DataFrame, string of column name, dictionary
    OUTPUT: original DataFrame with new columns
    '''

    new_df = df.copy()
    for idx in range(len(punctuation_dict)):
        col = pd.DataFrame(df[column].apply(count_character,
                           character=list(punctuation_dict.values())[idx]))
        col.columns = [list(punctuation_dict.keys())[idx]]
        new_df = pd.concat([new_df, col], axis=1)

    return new_df


def mention_hashtag_url(df, column):
    '''
    Takes a DataFrame and a specified column of tweetokenized tweets and
    creates new columns containing the count of @mentions, #hashtags, and URLs
    in the tweet
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with four new columns
    '''

    new_df = t.tweet_tokenize(df, 'text')
    new_df['mentions'] = new_df['tweetokenize'].apply(
                         lambda x: x.count('<USER>'))
    new_df['hashtags'] = new_df['tweetokenize'].apply(
                         lambda x: x.count('<HASHTAG>'))
    new_df['urls'] = new_df['tweetokenize'].apply(
                         lambda x: x.count('<URL>'))
    return new_df


def identify_quoted_retweet(text):
    '''
    Takes a string of text and returns 1 if the text begins with '"@' and a 0
    if not
    INPUT: string
    OUTPUT: int
    '''

    return (0 if re.match('^"@', text) is None else 1)


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


def all_caps(text):
    '''
    Takes a string of text and counts the number of ALL UPPERCASE words
    INPUT: string
    OUTPUT: int
    '''

    return (len(re.findall('\s([A-Z][A-Z]+)', text)))


def apply_all_caps(df, column):
    '''
    Takes a DataFrame and a specified column of text and creates a new column
    with the count of fully capitalized words in the text
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with one new column
    '''

    new_df = df.copy()
    new_df['all_caps'] = new_df[column].apply(all_caps)
    return new_df


def random_capitalization(df, column):
    '''
    Takes a DataFrame and a specified column of text and creates a new column
    with the count of randomly capitalized words in the text
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with one new column
    '''

    new_df = df.copy()
    exp = r"(?<!\.\s)(?<!\!\s)(?<!\?\s)\b[A-Z][a-z]*[^'][^I]\b"
    new_df['random_caps'] = new_df[column].apply(lambda x:
                                                 len(re.findall(exp, x)))
    return new_df


def mention_start(text):
    '''
    Takes a text string and outputs 1 if the string begins with "<USER>" and
    0 if not.
    INPUT: string
    OUTPUT: int
    '''
    return 1 if text[:6] == '<USER>' else 0
