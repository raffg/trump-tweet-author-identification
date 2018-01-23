# Trump Tweet Author Identification

## Background
In early December, Trump’s personal Twitter account tweeted:

![I had to fire General Flynn because he lied to the Vice President and the FBI. He has pled guilty to those lies. It is a shame because his actions during the transition were lawful. There was nothing to hide!](images/flynn_tweet.png)

Several legal experts argued that this tweet provided evidence that Trump obstructed justice. Trump defended himself by claiming that his lawyer John Dowd wrote and posted the tweet. But did he really?

Forensic text analysis was an early field in machine learning and has been used in cases as varied as identifying the unabomber to discovering J.K. Rowling as the true identity of the author Robert Galbraith to determining the specific authors of each of the Federalist Papers. This project is an effort to identify tweets on [@realDonaldTrump](https://twitter.com/realdonaldtrump) as written by Trump himself or by his staff when using his account.

Prior to March 26, 2017, Trump was tweeting using a Samsung Galaxy device while his staff were tweeting using an iPhone. From this information provided in the metadata of each tweet, we know whether it was Trump himself or his staff tweeting. After March however, Trump switched to using an iPhone as well, so identification of the tweeter cannot come from the metadata alone and must be deduced from the content of the tweet.

### Potential Tweeters

These individuals have been reported in the news as possible tweeters on Trump's Twitter account. The Start Date is the date their association with the Trump Campaign or Administration was announced, and the end date is when their positions were terminated.

|Name|Start Date|End Date|Twitter Handle|
|----|----------|--------|--------------|
|Donald Trump|2009-05-04|present|@realDonaldTrump|
|Sean Spicer|2016-12-22|2017-07-21|@seanspicer|
|Reince Priebus|2016-11-13|2017-07-27|@Reince|
|Steve Bannon|2016-08-17|2017-08-18|@SteveKBannon|
|Kellyanne Conway|2016-07-01|present|@KellyannePolls|
|Anthony Scaramucci|2017-07-21|2017-07-31|@Scaramucci|
|Dan Scavino|2015-06-01|present|@DanScavino|
|John Dowd|2017-07-16|present|N/A|


## Data

I used Brendan Brown's [Trump Tweet Data Archive](https://github.com/bpb27/trump_tweet_data_archive) to collect all tweets from the beginning of Trump's account in mid-2009 up until the end of 2017. This set consists of nearly 33,000 tweets. Even though I know from whose device a tweet originated, there is still some ambiguity around the authorship because Trump is known to dictate tweets to assistants, so a tweet may have Trump's characteristics but be posted from a non-Trump device, and also (especially during the campaign) to write tweets collaboratively with aides, making true authorship unclear.

## Feature engineering

### Style
I looked at the style of each tweet by counting various punctuation marks (the number of exclamation marks, for example), the number of /@mentions and #hashtags, and average tweet/sentence/word length.

### Trump quirks
I also created features for what I have recognized as Trump's rather unique Twitter behavior. These features include the "quoted retweet" (where Trump copies and pastes a another user's tweet onto his own timeline and surrounds it in quote marks), words written in ALL CAPS, and also middle-of-the-night tweeting.

### Sentiment
I used C.J. Hutto's [VADER](https://github.com/cjhutto/vaderSentiment) package to extract the sentiment of each tweet.VADER, which stands for Valence Aware Dictionary and sEntiment Reasoner, is a lexicon and rule-based tool that is specifically tuned to social media. Given a string of text, it outputs a number between 0 and 1 for negativity, positivity, and neutrality
for the text, as well as a compound score from -1 to 1 which is an aggregate measure.

### Emotion
The National Research Institute of Canada created a lexicon of over 14,000 words, each rated as belong to any of 10 emotion classes. For each tweet, I counted the number of words for each emotion class and assigned the tweet that count score for each emotion.

### Word choice
I performed TF-IDF on the text of each tweet in order to pick up vocabulary unique to Trump or his staff.

### Grammatical structure
I knew the phrasing of Trump's tweets would stand out from that of his staff, so in order to capture this I performed part-of-speech replacment on each tweet, reducing it to a string of its parts of speech. For example, the phrase "Hello. This is a tweet which has been parsed for parts of speech" would be replaced with "UH . DT BZ DT NN WDT VBZ VBN VBN IN NNS IN NN ", using the [Penn part of speech tags](https://cs.nyu.edu/grishman/jet/guide/PennPOS.html).


## Models

I created models for Naive Bayes, SVM, Logistic Regression with Ridge Regularization, KNN, and the ensemble methods of Random Forest and AdaBoost. All models achieved accuracy, precision, and recall rates in the low 90%s, except for Naive Bayes which was in the mid 80%s. For my final model, I found that an ensemble of these individual models worked best.

Additionally, I used the Ridge Regularization to iteratively drive each of the roughly 900 feature coefficients to zero with ever increasing alpha values. This allowed me to rank each feature in order of its importance to the logistic regression model.

## Results

One of the most interesting results from my analysis is the characteristics which identify a tweet as coming from Trump or from someone else. From my Ridge analysis, the top Trump features are:

* Quoted retweet
* @mentions
* Between 10pm and 10am
* Exclamations!!!
* ALL CAPS
* Tweet length: 114 characters
* @realDonaldTrump

The top features of non-Trump tweets are:

* True retweets
* The word “via”
* Between 10am and 4pm
* Semicolons
* Periods
* Tweet length: 103 characters
* @BarackObama

Trump's tweets are in general more emotive than his aides' tweets, exhibiting high scores for the emotions surprise, anger, negativity, disgust, joy, sadness, and fear. Non-Trump tweets, in contrast, are relatively unemotional, and feature many URLs, hashtags, and organization names.

## The Flynn Tweet

And as for that Flynn Tweet? My analysis strongly indicates it was written by Trump himself, not by his lawyer and they both claim. The Logistic Regression outputs a probability estimate of 97% that it came from Trump. Interestingly the [/@RPMMAS](https://twitter.com/RPMMAS) twitter account performed an informal poll of its users and received almost 2000 responses, with 96% indicating they believed the tweet to have come from Trump:

![WH claims his lawyer wrote this tweet: "I had to fire General Flynn because he lied to the Vice President and the FBI. He has pled guilty to those lies. It is a shame because his actions during the transition were lawful. There was nothing to hide!" Do you believe that's true?](images/flynn_tweet_poll.png)

A word of caution though: not all of my models individually agreed that Trump wrote it. Specifically, AdaBoost, KNN, and SVM indicated that it is a non-Trump tweet. Random Forest, Naive Bayes, and Logistic Regression all output Trump as the author. In my opinion, after reviewing thousands of Trump tweets throughout this project and evaluating all features which describe his tweets, I find the topic, sentiment, and emotion very much to be Trumpish, while the phrasing, grammar, and punctuation all indicate another author. I believe the tweet was written collaboratively, with Trump feeding someone the gist of the tweet and that unknown author actually composing it.


## Sources

*Many thanks to the following packages and lexicons!*

Trump's tweet data is from Brendan Brown's [Trump Tweet Data Archive](https://github.com/bpb27/trump_tweet_data_archive)

Trump aide data was scraped from Twitter using Ahmet Taspinar's [twitterscraper](https://github.com/taspinar/twitterscraper) with the query "twitterscraper 'from:twitter_handle since:2009-01-01 until:2017-12-31' -o scraped_tweets.json"

VADER sentiment analysis was performed using [C.J. Hutto's repo](https://github.com/cjhutto/vaderSentiment)

The National Research Institute of Canada kindly gave me access to the [NRC Word-Emotion Association Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)

Lastly, I used Jared Suttles' [Tweetokenize](https://github.com/jaredks/tweetokenize) to aid in my part-of-speech analysis
