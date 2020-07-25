# Senti API

## About
Senti API is a Python-based RESTful API server built using the popular Flask micro web framework. Senti API uses NLTK, a leading Natural Language Processing Python framework, to expose several sentiment analysis endpoints to clients.

Senti API currently provides two distinct sentiment analysis methods using different algorithms:  Naive Bayes classification and VADER (Valence Aware Dictionary and sEntiment Reasoner), as well as a bulk analysis endpoint for multiple documents using the VADER approach.
Senti API is hosted on [Heroku](https://senti-ment-api.herokuapp.com/) and is publicly accessible and key-free for the moment. Visit the API to browse the basic documentation generated using Swagger and flask-restx.  
Test out the API using Senti Playground, a web application designed for just for that, and has a host of other features.
Senti API is also backed by a AWS DynamoDB database which is currently used to persist our Gen Z corpora, which will be used for further training of our classifiers.

## Endpoints
```GET /bayes/{text}``` - This endpoint will return a sentiment classification for a requested string of text. Note that the Naive Bayes classifier does not produce polarity scores.
```GET /vader/{text}``` - This endpoint will return a sentiment classification and polarity scores for a requested string of text.
```POST /bulk``` - This endpoint will return the sentiment classification and polarity scores for a array of requested documents (text).
```POST /corpus``` - This endpoint will receive a phrase submission from a client and insert it into a DynamoDB table.
```GET /corpus``` - This endpoint will return the record count of a DynamoDB table. Note that counts are only updated every six hours as per the DynamoDB documentation.

## Run
Install the following Python packages using ```pip```
|Dependencies|
|------------|
re
requests
boto3
uuid
flask
flask_restx
nltk

You will also need to install the NLTK data packages listed in ```nltk.txt```
See this [documentation](https://www.nltk.org/data.html) for more information.

To run the server locally (```localhost:5000```), run the following command in terminal
```python3 wsgi.py```
This will start the server in development mode. Server logs will be printed to console.

## UCT DevSoc Competition Notes
Senti API is hosted on a free-plan Heroku dyno, and thus the server only spins-up on user request. The consequence of this is that an initial cold-start request is often much slower than subsequent API calls, so please do be patient. The API documentation is by no means comprehensive, and needs to still be expanded upon.
Secrets management for the AWS DynamoDB backend integration is acheived using Heroku Environment Variables.

Senti API primarily leverages Flask and NLTK to provide its functionality, and boto3 for AWS integration. NLTK data is downloaded into the production environment on Heroku. Although a pickled classifier is used for production to prevent constant training of the classifier on server startup, the source code for model training using the NLTK tweets dataset is also present in the repo for perusal. All the source code is documented comprehensively and confirms to PEP standards. Additional comments are also present when deemed essential.

I hope you find this API useful, and have fun using Senti Playground to try out this API, and explore other sentiment analysis features!
