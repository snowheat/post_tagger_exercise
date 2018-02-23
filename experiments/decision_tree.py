from sklearn import tree
from sklearn import preprocessing
import numpy as np


class DecisionTree:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, X_training_data):
        lb = preprocessing.LabelBinarizer()
        y = lb.fit_transform(X_training_data['labels'])
        self.__model = tree.DecisionTreeClassifier()
        self.__model = self.__model.fit(X_training_data['data_tfidf'], y)

        predicted_y = self.__model.predict(X_training_data['data_tfidf'])
        print(np.mean(predicted_y == y))
        pass

    def test(self, X_test_data):
        lb = preprocessing.LabelBinarizer()
        y = lb.fit_transform(X_test_data['labels'])
        predicted_y = self.__model.predict(X_test_data['data_tfidf'])

        print(np.mean(predicted_y == y))
        pass
