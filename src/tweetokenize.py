# Requires Python 2.7

from tweetokenize import Tokenizer


gettokens = Tokenizer(usernames='@USER', urls='<URL>', allcapskeep=True,
                      hashtags='#hashtag', times=',<time>', numbers='<number>')
gettokens.tokenize("@yankzpat: HEY! I hope to meet @realDonaldTrump on \
                   Thursday in Mason City, IA and get an autograph and a \
                   picture! Can't wait!! #Trump2016 http://t.co/v0n9ZF1ttS")
