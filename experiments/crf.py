from nltk.tag import CRFTagger


class CRF:
    def __init__(self):
        self.__model = type('test', (object,), {})()
        pass

    def train(self, X_training_data):
        self.__model = CRFTagger()
        self.__model.train(X_training_data, 'crf.model')
        pass

    def test(self, X_test_data):

        total = 0
        correct = 0
        for kalimat in X_test_data:
            temp = []
            for word in kalimat:
                temp.append(word[0])

            if len(temp) != 0:
                predicted_y = self.__model.tag(temp)
                for i in range(len(predicted_y)):
                    total += 1
                    if predicted_y[i][1] == kalimat[i][1]:
                        correct += 1

        print(correct, total)
        print(correct / total)
    pass
