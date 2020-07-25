import re
import string
import random
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
Script that trains a Naive Bayes sentiment classifier using a Twitter dataset
from NLTK Data, and pickles the trained classifier for use by Senti API.
"""

def remove_noise(tweet_tokens, stop_words=()):
    """
    Removes noise from a list of text tokens using regular expressions
    and lemmatization
    Parameters
    ----------
    tokens (list): Text tokens
    stop_words (tuple): stopwords
    Returns
    -------
    List
    """
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"
            "(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            token,
        )
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        #Part-of-speech
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"
        #Token lemmatization
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        #Token cleaning
        if (
                len(token) > 0
                and token not in string.punctuation
                and token.lower() not in stop_words
        ):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    """
    Generates a list of words from all tweet tokens.
    Parameters
    ----------
    cleaned_tokens_list (list): Cleaned tweet tokens
    Returns
    -------
    Generator
    """
    for tokens in cleaned_tokens_list:
        for token in tokens:
            #Generator function
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    """
    Generates a dictionary from cleaned tokens.
    Parameters
    ----------
    cleaned_tokens_list (list): Cleaned tweet tokens
    Returns
    -------
    Generator
    """
    for tweet_tokens in cleaned_tokens_list:
        #Generator function
        yield dict([token, True] for token in tweet_tokens)


def train_model():
    """
    Trains a Naive Bayes sentiment classifier using the twitter_samples
    dataset from NLTK. Each tweet is tokenized and cleaned to produce a training
    dataset for the machine learning model.
    Parameters
    ----------
    Returns
    -------
    NaiveBayesClassifier
    """
    #Load dataset from nltk data
    positive_tweets = twitter_samples.strings("positive_tweets.json")
    negative_tweets = twitter_samples.strings("negative_tweets.json")

    #Retrieve english stop words
    stop_words = stopwords.words("english")

    #Tweet tokenization
    positive_tweet_tokens = twitter_samples.tokenized("positive_tweets.json")
    negative_tweet_tokens = twitter_samples.tokenized("negative_tweets.json")

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    #Token cleaning
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    #Extract words from tokens
    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    #Frequency distribition of words
    freq_dist_pos = FreqDist(all_pos_words)

    positive_tokens_for_model = get_tweets_for_model(
        positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(
        negative_cleaned_tokens_list)

    #Create datasets
    positive_dataset = [
        (tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model
    ]

    negative_dataset = [
        (tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model
    ]
    #Merge individual datasets into singular training data
    dataset = positive_dataset + negative_dataset

    train_data = dataset

    classifier = NaiveBayesClassifier.train(train_data)
    return classifier

def main():
    """
    Runs training for the Naive Bayes classifier and pickles the trained model
    """
    classifier = train_model()
    save_classifier = open("naivebayes.pickle","wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

if __name__ == "__main__":
    main()
