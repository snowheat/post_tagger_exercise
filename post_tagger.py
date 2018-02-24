from experiments.hmm import HMM
from experiments.crf import CRF

import timeit
import pickle
from experiments.knn import Knn
from experiments.sgd import SGD
from experiments.rule_based import RuleBased
from experiments.svm import Svm
from experiments.neural_net import NeuralNet
from experiments.random_forest import RandomForest
from experiments.decision_tree import DecisionTree
from experiments.naive_bayes import NaiveBayes
from experiments.vector_space_model import VectorSpaceModel

import numpy as np
import numpy.matlib

class PostTagger:

    def __init__(self, algorithm, training_parameters={}, testing_parameters={}):

        self.__dataset_basic = []
        self.__training_parameters = training_parameters
        self.__testing_parameters = testing_parameters
        self.__dataset_hmm = []

        self.__corpus = self.__get_corpus()
        self.__training_data, self.__training_feature_array, self.__training_feature_ranges, self.__training_labels = self.__get_training_data_non_hmm(self.__corpus)
        self.__testing_data, self.__testing_labels = self.__get_testing_data_non_hmm(self.__corpus)


        np.set_printoptions(threshold=np.nan)
        #print(self.__corpus[901])

        self.__experiment = {
            'rule_based': RuleBased(),  # insan
            'decision_tree': DecisionTree(),  # sigit
            'naive_bayes': NaiveBayes(),
            'random_forest': RandomForest(),
            'svm': Svm(),  # insan
            'neural_network': NeuralNet(),
            'vector_space_model': VectorSpaceModel(),
            'knn': Knn(),
            'sgd': SGD(),
        }[algorithm]


    def __get_corpus(self):
        with open('id_pud-ud-test.conllu', 'r', encoding='utf8') as fp:

            all_text = fp.read().lower().split('\n\n')

        fp.close()

        corpus = []


        for block in all_text:

            row = block.split('\n')

            row_data = []

            for term in row[2:]:
                # print(term)
                term_part = term.split()

                row_data.append([term_part[1], term_part[3]])

            corpus.append(row_data)

            self.__dataset_basic.append(row_data)

        # print(self.__dataset_basic[901:])
        # self.__HMM = HMM()
        # self.__train()
        # self.__test()

        return corpus


    def __get_train_data(self):
        train_data = {}


        return train_data



    def __get_test_data(self):
        pass

    def train(self):
        start = timeit.default_timer()
        #self.__HMM.train(self.__dataset_basic[:900])
        self.__experiment.train({
            'data': self.__training_data,
            'features': self.__training_feature_array,
            'labels': self.__training_labels,
        })
        stop = timeit.default_timer()
        print("training time : ", stop - start)

    def test(self):
        start = timeit.default_timer()
        #self.__HMM.test(self.__dataset_basic[901:])
        self.__experiment.test({
            'data': self.__testing_data,
            'features': self.__training_feature_array,
            'labels': self.__testing_labels,
        })
        stop = timeit.default_timer()
        print("testing time : ", stop - start)

    def __get_training_data_non_hmm(self, corpus):

        unique_postags = []
        unique_terms = []
        words_number = 0

        raw_training_data = corpus[:900]

        for text in raw_training_data:
            for index, word_data in enumerate(text):
                if word_data[1] not in unique_postags:
                    unique_postags.append(word_data[1])
                if word_data[0] not in unique_terms:
                    unique_terms.append(word_data[0])
                words_number+=1

        print(unique_postags)
        # print(len(unique_postags))

        feature_array = np.append(unique_postags, unique_terms)
        feature_array = np.append(feature_array, unique_terms)
        feature_array = np.append(feature_array, unique_terms)

        feature_ranges = {
            'postags_before_start' : 0,
            'postags_before_end' : len(unique_postags),
            'words_before_start' : len(unique_postags),
            'words_before_end' : len(unique_postags) + len(unique_terms),
            'words_middle_start': len(unique_postags) + len(unique_terms),
            'words_middle_end': len(unique_postags) + len(unique_terms) + len(unique_terms),
            'words_after_start': len(unique_postags) + len(unique_terms) + len(unique_terms),
            'words_after_end': len(unique_postags) + len(unique_terms) + len(unique_terms) + len(unique_terms),
        }

        training_data = np.matlib.zeros((words_number, feature_array.size))
        training_labels = np.full(words_number, 1, dtype='str')

        # index = np.where(feature_array[ feature_ranges['words_before_start']:feature_ranges['words_before_end'] ] == 'damai')

        i=0
        #np.set_printoptions(threshold=np.nan)


        # each text in training data
        for text in raw_training_data:

            # each word in text
            for index, word_data in enumerate(text):

                status = 'middle_word'

                idx_label = np.where(feature_array[feature_ranges['postags_before_start']:feature_ranges['postags_before_end']] == text[index][1])[0][0]
                training_labels[i] = str(idx_label)

                # first word in text
                if index == 0:
                    status = 'first_word'

                # last word in text
                elif index == len(text) - 1:
                    status = 'last_word'



                if status == 'last_word' or status == 'middle_word':

                    # insert into postag columns group on training data
                    idx_postag = np.where(feature_array[feature_ranges['postags_before_start']:feature_ranges['postags_before_end']] == text[index - 1][1])[0][0]
                    training_data[i, feature_ranges['postags_before_start'] + idx_postag] = 1.0

                    # insert into words_before columns group on training data
                    idx_word_before = np.where(feature_array[feature_ranges['words_before_start']:feature_ranges['words_before_end']] == text[index - 1][0])[0][0]
                    training_data[i, feature_ranges['words_before_start'] + idx_word_before] = 1.0


                # insert into words_middle columns group on training data
                idx_word_middle = np.where(feature_array[feature_ranges['words_middle_start']:feature_ranges['words_middle_end']] == text[index][0])[0][0]
                training_data[i, feature_ranges['words_middle_start'] + idx_word_middle] = 1.0


                if status == 'first_word' or status == 'middle_word':
                    # insert into words_after columns group on training data
                    idx_word_after = np.where(feature_array[feature_ranges['words_after_start']:feature_ranges['words_after_end']] == text[index + 1][0])[0][0]
                    training_data[i, feature_ranges['words_after_start'] + idx_word_after] = 1.0


                    """
                    print(text[index][0], idx_word_after, feature_ranges['words_after_start'] + idx_word_after,
                          feature_array[feature_ranges['words_after_start'] + idx_word_after])
                    """

                """
                if i == 1:
                    print(training_data[i])
                    print(training_data[i, feature_ranges['words_after_start'] + idx_word_after])
                    print(feature_array[feature_ranges['words_after_start'] + idx_word_after])
                """

                i+=1


        #np.set_printoptions(threshold=np.nan)

        """
        try:
            feature_array = pickle.load(open('feature_array.pickle', 'rb'))
        except (OSError, IOError) as e:
            pickle.dump(feature_array, open('feature_array.pickle', 'wb'))
        """

        return training_data, feature_array, feature_ranges, training_labels

    def __get_testing_data_non_hmm(self, corpus):

        raw_testing_data = corpus[901:]

        training_data = self.__training_data
        feature_array = self.__training_feature_array
        feature_ranges = self.__training_feature_ranges
        oov_array = []
        total_oov = 0

        words_number = 0

        for text in raw_testing_data:
            for index, word_data in enumerate(text):
                words_number += 1

        testing_data = np.matlib.zeros((words_number, feature_array.size))
        testing_labels = np.full(words_number, 1, dtype='str')

        i = 0
        # np.set_printoptions(threshold=np.nan)

        # each text in training data
        for text in raw_testing_data:

            # each word in text
            for index, word_data in enumerate(text):


                status = 'middle_word'

                idx_label = np.where(
                    feature_array[feature_ranges['postags_before_start']:feature_ranges['postags_before_end']] ==
                    text[index][1])[0][0]
                testing_labels[i] = str(idx_label)


                # first word in text
                if index == 0:
                    status = 'first_word'

                # last word in text
                elif index == len(text) - 1:
                    status = 'last_word'

                if status == 'last_word' or status == 'middle_word':
                    # insert into postag columns group on training data
                    idx_postag = np.where(
                        feature_array[feature_ranges['postags_before_start']:feature_ranges['postags_before_end']] ==
                        text[index - 1][1])[0][0]
                    testing_data[i, feature_ranges['postags_before_start'] + idx_postag] = 1.0

                    # insert into words_before columns group on training data
                    try:
                        idx_word_before = np.where(
                            feature_array[feature_ranges['words_before_start']:feature_ranges['words_before_end']] ==
                            text[index - 1][0])[0][0]
                        testing_data[i, feature_ranges['words_before_start'] + idx_word_before] = 1.0
                    except Exception:
                        if text[index - 1][0] not in oov_array:
                            oov_array.append(text[index - 1][0])
                            total_oov += 1

                # insert into words_middle columns group on training data
                try:

                    idx_word_middle = np.where(
                        feature_array[feature_ranges['words_middle_start']:feature_ranges['words_middle_end']] ==
                        text[index][0])[0][0]
                    testing_data[i, feature_ranges['words_middle_start'] + idx_word_middle] = 1.0

                except Exception:
                    if text[index][0] not in oov_array:
                        oov_array.append(text[index][0])
                        total_oov += 1

                if status == 'first_word' or status == 'middle_word':

                    try:
                        # insert into words_after columns group on training data
                        idx_word_after = np.where(
                            feature_array[feature_ranges['words_after_start']:feature_ranges['words_after_end']] ==
                            text[index + 1][0])[0][0]
                        testing_data[i, feature_ranges['words_after_start'] + idx_word_after] = 1.0
                    except Exception:
                        if text[index + 1][0] not in oov_array:
                            oov_array.append(text[index + 1][0])
                            total_oov += 1


                i += 1

        return testing_data, testing_labels
