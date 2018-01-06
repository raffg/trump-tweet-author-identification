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


def count_mentions(text):
    '''
    counts the number of @mentions in a string of text
    INPUT: string
    OUTPUT: int
    '''

    return count_character(text, '@')


def count_hashtags(text):
    '''
    counts the number of #hashtags in a string of text
    INPUT: string
    OUTPUT: int
    '''

    return count_character(text, '#')


def count_urls(text):
    '''
    counts the number of urls in a string of text
    INPUT: string
    OUTPUT: int
    '''

    return count_character(text, '://')


def mention_hashtag_url_columns(df, column):
    '''
    takes a DataFrame and a column of text and creates new columns containing
    the number of occurances of @mentions, #hashtags, and urls by row
    INPUT: DataFrame, string of column name
    OUTPUT: original DataFrame with three new columns
    '''

    mentions = pd.DataFrame(df['text'].apply(count_mentions))
    mentions.columns = ['mentions']

    hashtags = pd.DataFrame(df['text'].apply(count_hashtags))
    hashtags.columns = ['hashtags']

    urls = pd.DataFrame(df['text'].apply(count_urls))
    urls.columns = ['urls']

    return pd.concat([df, mentions, hashtags, urls], axis=1)
