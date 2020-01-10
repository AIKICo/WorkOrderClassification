from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas
from hazm import *


def notmalizetext(text):
    normalize = Normalizer()
    return normalize.normalize(text)


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
        for vectorizer in vectorizers:
            string = ''
            string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

            # train
            vectorize_text = vectorizer.fit_transform(train_data.v2)
            classifier.fit(vectorize_text, train_data.v1)

            # score
            vectorize_text = vectorizer.transform(test_data.v2)
            score = classifier.score(vectorize_text, test_data.v1)
            string += '. Has score: ' + str(score)
            print(string)


if __name__ == '__main__':
    data = pandas.read_csv('workorder.csv', encoding='utf-8')
    data['v2'].apply(notmalizetext)
    learn = data[:12388]
    test = data[12388:]
    perform(
        [
            # BernoulliNB(),
            # GaussianNB(),
            # RandomForestClassifier(n_estimators=100, n_jobs=-1),
            # AdaBoostClassifier(),
            # BaggingClassifier(),
            # ExtraTreesClassifier(),
            # GradientBoostingClassifier(),
            # DecisionTreeClassifier(),
            # DummyClassifier(),
            # PassiveAggressiveClassifier(),
            # RidgeClassifier(),
            # RidgeClassifierCV(),
            # SGDClassifier(),
            # OneVsRestClassifier(SVC(kernel='linear')),
            OneVsRestClassifier(LogisticRegression()),
            # KNeighborsClassifier()
        ],
        [
            # CountVectorizer(),
            TfidfVectorizer(),
            # HashingVectorizer()
        ],
        learn,
        test
    )
