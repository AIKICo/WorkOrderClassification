import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import pickle
from hazm import *

app = Flask(__name__)
global Classifier
global Vectorizer


def notmalizetext(text):
    normalize = Normalizer()
    return normalize.normalize(text)


with open('SVC.pkl', 'rb') as f:
    Classifier = pickle.load(f)

Vectorizer = TfidfVectorizer()
data = pandas.read_csv('workorder.csv', encoding='utf-8')
data['v2'].apply(notmalizetext)
train_data = data[:11071]
vectorize_text = Vectorizer.fit_transform(train_data.v2)


@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    error = ''
    predict_proba = ''
    predict = ''

    global Classifier
    global Vectorizer
    try:
        if len(message) > 0:
            vectorize_message = Vectorizer.transform([message])
            predict = Classifier.predict(vectorize_message)[0]
            predict_proba = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    return jsonify(
        message=message, predict_proba=predict_proba,
        predict=predict, error=error)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
