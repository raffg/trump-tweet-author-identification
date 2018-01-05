import re


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
