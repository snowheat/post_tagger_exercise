import operator


class RuleBased:

    def __init__(self):
        self.__model = type('test', (object,), {})()

        pass

    def train(self, X_training_data):
        self.__model = X_training_data['spam_words']
        self.__model_test(X_training_data)

        pass

    def test(self, X_test_data):
        self.__model_test(X_test_data)

        pass

    def __model_test(self, test_data):

        total_spam_in_testing_data = 0
        total_spam_detected_as_spam = 0

        for index, row in enumerate(test_data['data']):
            row_split = row.split()
            spam_terms = 0
            spam = False

            for term in row_split:
                if term in self.__model:
                    spam_terms += 1

            if spam_terms/(len(row_split)+1) * 100 > 75:
                spam = True

            if test_data['labels'][index] == 'spam':
                total_spam_in_testing_data += 1
                if spam == True:
                    total_spam_detected_as_spam += 1



            #print(index, test_data['labels'][index], spam_terms, "/", len(row_split), row)

        print(total_spam_detected_as_spam/total_spam_in_testing_data)