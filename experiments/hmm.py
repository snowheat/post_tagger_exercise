# src: https://github.com/hmmlearn/hmmlearn
# tutorial: http://hmmlearn.readthedocs.io/en/latest/tutorial.html

# from hmmlearn import hmm
from nltk import hmm


class HMM:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, X_training_data):
        from nltk.probability import ELEProbDist
        estimator = lambda fdist, bins:ELEProbDist(fdist)
        self.__model = hmm.HiddenMarkovModelTrainer()
        self.__tagger = self.__model.train_supervised(X_training_data, estimator=estimator)

        # predicted_y = self.__model.predict(X_training_data['data_tfidf'])
        # print(np.mean(predicted_y == X_training_data['labels']))
        pass

    def test(self, X_test_data):

        total = 0
        correct = 0
        for kalimat in X_test_data:
            # print(kalimat)
            temp = []
            for word in kalimat:
                temp.append(word[0])

            if len(temp) != 0:
                predicted_y = self.__tagger.tag(temp)
                # print(predicted_y)
                for i in range(len(predicted_y)):
                    total += 1
                    if predicted_y[i][1] == kalimat[i][1]:
                        correct += 1
                # print(predicted_y)
                # predicted_y = self.__tagger.tag(kalimat)
                # print(predicted_y)

        print(correct,total)
        print(correct/total)
    # print(np.mean(predicted_y == X_test_data['labels']))
    pass
