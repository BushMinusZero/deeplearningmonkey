import nltk
from nltk.corpus import twitter_samples

nltk.download('twitter_samples')

print("Files: ", twitter_samples.fileids())

tweets = twitter_samples.strings('tweets.20150430-223406.json')
print("Total tweets: ", len(tweets))
for tweet in tweets[:10]:
    print(tweet)
