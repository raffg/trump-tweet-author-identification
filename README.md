# Trump Tweet Author Identification

In early December, Trump’s personal Twitter account tweeted

>“I had to fire General Flynn because he lied to the Vice President and the FBI. He has pled guilty to those lies. It is a shame because his actions during the transition were lawful. There was nothing to hide!”

Several legal experts argued that this tweet provided evidence that Trump obstructed justice. Trump defended himself by claiming that his lawyer John Dowd wrote and posted the tweet. But did he really?

Forensic text analysis was an early field in machine learning and has been used in cases as varied as identifying the unabomber to identifying J.K. Rowling as the true author of a pseudonymous novel. This project is an effort to identify tweets on [@realDonaldTrump](https://twitter.com/realdonaldtrump) as written by Trump himself or by his staff when using his account.

Prior to March 2017, Trump was tweeting using a Samsung Galaxy device while his staff were tweeting using an iPhone. From this information provided in the metadata of each tweet, we know whether it was Trump himself or his staff tweeting. After March however, Trump switched to using an iPhone as well, so identification of the tweeter cannot come from the metadata alone and must be deduced from the content of the tweet.

### Potential Tweeters

These individuals have been reported in the news as possible tweeters on Trump's Twitter account. The Start Date is the date their association with the Trump Campaign or Administration was announced, and the end date is the date when Trump began using an iPhone device himself, making explicit identification impossible.

|Name|Start Date|End Date|Twitter Handle|
|----|----------|--------|--------------|
|Donald Trump|2009-05-04|2017-03-25|@realDonaldTrump|
|Sean Spicer|2016-12-22|2017-03-25|@seanspicer|
|Reince Priebus|2016-11-13|2017-03-25|@Reince|
|Steve Bannon|2016-08-17|2017-03-25|@SteveKBannon|
|Kellyanne Conway|2016-07-01|2017-03-25|@KellyannePolls|
|Dan Scavino|2015-06-01|2017-03-25|@DanScavino|
|John Dowd|N/A|N/A|N/A|



### Sources

Data scraped from Twitter using Ahmet Taspinar's [twitterscraper](https://github.com/taspinar/twitterscraper) with the query "twitterscraper 'from:realDonaldTrump since:2009-01-01 until:2017-12-31' -o scraped_tweets.json"

Master and Condensed data from Brendan Brown's [Trump Tweet Data Archive](https://github.com/bpb27/trump_tweet_data_archive)
