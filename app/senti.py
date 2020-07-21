import re
import string
import random
import requests
import pickle
from flask import Flask, request, jsonify, make_response
from flask_restx import Api, Resource, fields
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
api = Api(app=app, version="1.0", title="Senti API")

model = api.model(
    "Senti Model", {"text": fields.String(
        required=True, description="Text to classify")}
)

def remove_noise(tweet_tokens, stop_words=()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"
            "(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            token,
        )
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if (
                len(token) > 0
                and token not in string.punctuation
                and token.lower() not in stop_words
        ):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def naive_bayes(text, classifier):

    tokens = []
    stop_words = stopwords.words("english")

    text_tokens = remove_noise(word_tokenize(text))
    classification = classifier.classify(
        dict([token, True] for token in text_tokens)
    )
    return classification

def vader(text, classifier):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    print(score["compound"])
    if score["compound"] > 0.05 :
        score["classification"] = "Positive"
    elif score["compound"] < -0.05:
        score["classification"] = "Negative"
    else:
        score["classification"] = "Neutral"
    return score

saved_classifier = open("./app/naivebayes.pickle", "rb")
classifier = pickle.load(saved_classifier)
saved_classifier.close()
print("Sentiment Classifier Loaded")

@api.route("/vader/<string:text>")
class Vader(Resource):
    def options(self, text):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    @api.expect(model)
    def get(self, text):
        try:
            classification = vader(text, classifier)
            response = jsonify(
                {
                    "statusCode": 200,
                    "status": "Successful",
                    "classification": classification,
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        except Exception as error:
            return jsonify({"statusCode": 500, "status": "Error", "error": str(error)})

@api.route("/bayes/<string:text>")
class NaiveBayes(Resource):
    def options(self, text):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
        
    @api.expect(model)
    def get(self, text):
        try:
            result = naive_bayes(text, classifier)
            classification = {
                "classification": result,
            }
            response = jsonify(
                {
                    "statusCode": 200,
                    "status": "Successful",
                    "classification": classification,
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        except Exception as error:
            return jsonify({"statusCode": 500, "status": "Error", "error": str(error)})