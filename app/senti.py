import re
import string
import random
import requests
import pickle
import logging
import boto3
import uuid
import os
from datetime import datetime
from flask import Flask, request, jsonify, make_response
from flask_restx import Api, Resource, fields
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
Senti API for Sentiment Analysis built with Flask and NLTK for the NLP.
Provides endpoints for analysis, bulk sentiment analysis, and open-source
corpus contributions.

Classes:
    Item
    Document
    Corpus
    Bayes
    Vader
    Bulk
"""

# Create Flask application
app = Flask(__name__)
api = Api(app=app, version="1.0", title="Senti API")

# Logging Config
logging.basicConfig(format='%(asctime)s - [%(levelname)s] %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

# API Models
sentiment_analysis_resource = api.model(
    "Sentiment Analysis Resource", {"text": fields.String(
        required=True, description="Text to classify")}
)

sentiment_analysis_response = api.model("Sentiment Analysis Response", {
    "classification": fields.String(required=True, description="Sentiment classification"),
    "pos": fields.Float(required=True, description="Positive score"),
    "neg": fields.Float(required=True, description="Negative score"),
    "neu": fields.Float(required=True, description="Neutral score")
})

bayes_analysis_response = api.model("Bayes Sentiment Analysis Response", {
    "classification": fields.String(required=True, description="Sentiment classification"),
})

corpus_resource = api.model('Corpus Resource', {
    'user': fields.String(required=True, description="Authenticated user"),
    'phrase': fields.String(required=True, description="User phrase"),
    'sentiment': fields.String(required=True, description="User sentiment"),
})

corpus_response = api.model('Corpus Response', {
    'count': fields.Integer(required=True, description="Table record count"),
})

bulk_resource_document = api.model('Bulk Resource', {
    'text': fields.String(required=True, description="Document to classify"),
})


class Item(fields.Raw):
    """
    A class that represents a Corpus POST request payload.
    ...
    Attributes
    ----------
    value: dict
        request object
    Methods
    -------
    format(self, value):
        Returns a dictionary of attributes.
    """

    def format(self, value):
        return {'user': value.user, 'phrase': value.phrase, 'sentiment': value.sentiment}


class Document(fields.Raw):
    """
    A class that represents a Bulk POST request payload.
    ...
    Attributes
    ----------
    value: dict
        request object
    Methods
    -------
    format(self, value):
        Returns a dictionary of attributes.
    """

    def format(self, value):
        return {'text': value.text}


def remove_noise(tokens, stop_words=()):
    """
    Removes noise from a list of text tokens using regular expressions
    and lemmatization.
    Parameters
    ----------
    tokens (list): Text tokens
    stop_words (tuple): stopwords
    Returns
    -------
    List
    """
    cleaned_tokens = []
    for token, tag in pos_tag(tokens):
        token = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"
            "(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            token,
        )
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        # Part-of-speech
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
    """
    Classifies text using a pre-trained Naive Bayes classifier and returns
    a sentiment classification.
    Parameters
    ----------
    text (String): Text to classify
    classifier (Object): Trained naive bayes classifier
    Returns
    -------
    String
    """
    # Token cleaning
    text_tokens = remove_noise(word_tokenize(text))
    classification = classifier.classify(
        dict([token, True] for token in text_tokens)
    )
    return classification


def vader(text):
    """
    Classifies text using a SentimentIntensityAnalyzer trained using the
    VADER algorithm and returns a sentiment classification and polarity scores.
    Parameters
    ----------
    text (String): Text to classify
    classifier (Object): Trained naive bayes classifier
    Returns
    -------
    dict
    """
    # Create SentimentIntensityAnalyzer (using VADER) object
    sid = SentimentIntensityAnalyzer()
    # Analyse text
    score = sid.polarity_scores(text)
    # Convert float value to String for client-side rendering
    if score["compound"] > 0.05:
        score["classification"] = "Positive"
    elif score["compound"] < -0.05:
        score["classification"] = "Negative"
    else:
        score["classification"] = "Neutral"
    return score


def insert(item):
    """
    Inserts and open-source Gen Z corpus submission into a DynamoDB table instance.
    Parameters
    ----------
    item (dict): Corpus record to insert
    Returns
    -------
    None
    """
    dynamodb = boto3.resource('dynamodb',
                              region_name='us-east-1',
                              aws_access_key_id=os.environ['DYNAMODB_KEY'],
                              aws_secret_access_key=os.environ['DYNAMODB_SECRET'])
    id = str(uuid.uuid1())
    dt = str(datetime.utcnow())
    # DynamoDB put_item operation
    table = dynamodb.Table('senti-corpus')
    response = table.put_item(
        Item={
            'uuid': id,
            'datetime': dt,
            'user': item['user'],
            'phrase': item['phrase'],
            'sentiment': item['sentiment'],

        }
    )
    logging.info("Inserted item into DynamoDB table")
    return response


def item_count():
    """
    Retrieves the approximate record count from a DynamoDB table.
    Parameters
    ----------
    Returns
    -------
    int
    """
    dynamodb = boto3.resource('dynamodb',
                              region_name='us-east-1',
                              aws_access_key_id=os.environ['DYNAMODB_KEY'],
                              aws_secret_access_key=os.environ['DYNAMODB_SECRET'])
    table = dynamodb.Table('senti-corpus')
    return table.item_count


# Loads pre-trained classifier from .pickle file
saved_classifier = open("./app/naivebayes.pickle", "rb")
classifier = pickle.load(saved_classifier)
saved_classifier.close()
logging.info("Sentiment Classifier Loaded")


@api.route("/corpus")
@api.header("Access-Control-Allow-Origin", "*")
class Corpus(Resource):
    """
    A class that represents the /corpus API endpoint.
    ...
    Attributes
    ----------
    Methods
    -------
    options(self):
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS).
    post(self):
        Recevies a payload (phrase-sentiment submission) from a client
        and inserts it into the DynamoDB Table.
    get(self):
        Returns a response containing the record count of the DynamoDB Table.
    """
    @api.response(200, 'Success', headers={"Access-Control-Allow-Origin": "*",
                                           "Access-Control-Allow-Headers": "*",
                                           "Access-Control-Allow-Methods": "GET, POST, OPTIONS"})
    def options(self):
        """
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS)
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add(
            "Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    @api.expect(corpus_resource)
    @api.response(200, 'Success')
    @api.response(500, 'Server Error')
    def post(self):
        """
        Recevies a payload (phrase-sentiment submission) from a client
        and inserts it into the DynamoDB Table.
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        try:
            item = request.get_json()
            insert(item)
            response = jsonify(
                {
                    "statusCode": 200,
                    "status": "Successful",
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        except Exception as error:
            return jsonify({"statusCode": 500, "status": "Error", "error": str(error)})

    @api.response(200, 'Success', corpus_response)
    @api.response(500, 'Server Error')
    def get(self):
        """
        Returns a response containing the record count of the DynamoDB Table.
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        try:
            count = item_count()
            response = jsonify(
                {
                    "statusCode": 200,
                    "status": "Successful",
                    "count": count
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        except Exception as error:
            return jsonify({"statusCode": 500, "status": "Error", "error": str(error)})


@api.route("/vader/<string:text>")
@api.header("Access-Control-Allow-Origin", "*")
class Vader(Resource):
    """
    A class that represents the /vader API endpoint.
    ...
    Attributes
    ----------
    Methods
    -------
    options(self, text):
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS).
    get(self, text):
        Returns a response containing the classification and polarity scores
        for a particular text string using the VADER algorithm.
    """
    @api.response(200, 'Success', headers={"Access-Control-Allow-Origin": "*",
                                           "Access-Control-Allow-Headers": "*",
                                           "Access-Control-Allow-Methods": "GET, OPTIONS"})
    def options(self, text):
        """
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS).
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    @api.expect(sentiment_analysis_resource)
    @api.response(200, 'Success', sentiment_analysis_response)
    @api.response(500, 'Server Error')
    def get(self, text):
        """
        Returns a response containing the classification and polarity scores
        for a particular text string using the VADER algorithm.
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        try:
            classification = vader(text)
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
@api.header("Access-Control-Allow-Origin", "*")
class NaiveBayes(Resource):
    """
    A class that represents the /bayes API endpoint.
    ...
    Attributes
    ----------
    Methods
    -------
    options(self, text):
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS).
    get(self, text):
        Returns a response containing the sentiment classification
        for a particular text string using a pre-trained naive bayes classifier.
    """
    @api.response(200, 'Success', headers={"Access-Control-Allow-Origin": "*",
                                           "Access-Control-Allow-Headers": "*",
                                           "Access-Control-Allow-Methods": "GET, OPTIONS"})
    def options(self, text):
        """
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS).
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    @api.expect(sentiment_analysis_resource)
    @api.response(200, 'Success', bayes_analysis_response)
    @api.response(500, 'Server Error')
    def get(self, text):
        """
        Returns a response containing the sentiment classification
        for a particular text string using a pre-trained naive bayes classifier.
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
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


@api.route("/bulk")
@api.header("Access-Control-Allow-Origin", "*")
class Bulk(Resource):
    """
    A class that represents the /bulk API endpoint.
    ...
    Attributes
    ----------
    Methods
    -------
    options(self):
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS).
    post(self):
        Receives an array of documents and analyses each document's contents
        using the VADER algorithm, returning the classification and polarity scores
        for each requested document.
    """
    @api.response(200, 'Success', headers={"Access-Control-Allow-Origin": "*",
                                           "Access-Control-Allow-Headers": "*",
                                           "Access-Control-Allow-Methods": "POST, OPTIONS"})
    def options(self):
        """
        Returns a response with HTTP headers to a pre-flight request
        to allow for Cross-Origin Resource Sharing (CORS).
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    @api.expect([bulk_resource_document])
    @api.response(200, 'Success', [sentiment_analysis_response])
    @api.response(500, 'Server Error')
    def post(self):
        """
        Receives an array of documents and analyses each document's contents
        using the VADER algorithm, returning the classification and polarity scores
        for each requested document.
        Parameters
        ----------
        Returns
        -------
        JSON Object
        """
        try:
            payload = request.get_json()
            classification = []
            # Processing of each document
            for document in payload['documents']:
                classification.append(vader(document['text']))
            response = jsonify(
                {
                    "statusCode": 200,
                    "status": "Successful",
                    "bulkClassification": classification,
                }
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        except Exception as error:
            return jsonify({"statusCode": 500, "status": "Error", "error": str(error)})
