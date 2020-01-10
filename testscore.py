from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv
from hazm import *


def notmalizetext(text):
    normalize = Normalizer()
    return normalize.normalize(text)


if __name__ == '__main__':
    data = pandas.read_csv('workorder.csv', encoding='utf-8')
    data['v2'].apply(notmalizetext)
    train_data = data[:12388]
    test_data = data[12388:]

    classifier = OneVsRestClassifier(LogisticRegression())
    vectorizer = TfidfVectorizer()

    # train
    vectorize_text = vectorizer.fit_transform(train_data.v2)
    classifier.fit(vectorize_text, train_data.v1)

    csv_arr = []
    for index, row in test_data.iterrows():
        answer = row[0]
        text = row[1]
        vectorize_text = vectorizer.transform([text])
        predict = classifier.predict(vectorize_text)[0]
        if predict == answer:
            result = 'right'
        else:
            result = 'wrong'
        csv_arr.append([len(csv_arr), text, answer, predict, result])

        with open('test_score.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['#', 'text', 'answer', 'predict', result])

            for ro in csv_arr:
                spamwriter.writerow(ro)