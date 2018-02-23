from sklearn import naive_bayes
import numpy as np
from scipy.sparse.linalg import svds

class NaiveBayes:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, X_training_data):
        self.__u, self.__s, self.__vt = svds(X_training_data['data_tfidf'].transpose())

        self.__model = naive_bayes.GaussianNB()
        self.__model = self.__model.fit(self.__vt.transpose(), X_training_data['labels'])

        # self.__model = self.__model.fit(X_training_data['data_tfidf'], X_training_data['labels'])

        # predicted_y = self.__model.predict(X_training_data['data_tfidf'])
        # print(np.mean(predicted_y == X_training_data['labels']))
        pass

    def test(self, X_test_data):
        # predicted_y = self.__model.predict(X_test_data['data_tfidf'])

        test_data = np.dot(np.dot(X_test_data['data_tfidf'], self.__u), np.linalg.inv(np.diag(self.__s)))
        predicted_y = self.__model.predict(test_data)

        print(np.mean(predicted_y == X_test_data['labels']))
        pass
