from experiments.hmm import HMM
from experiments.crf import CRF


class PostTagger:
    def __init__(self, training_parameters={}, testing_parameters={}):

        self.__training_parameters = training_parameters
        self.__testing_parameters = testing_parameters
        self.__dataset_hmm = []

        self.__corpus = self.__get_corpus();
        self.__dataset_non_hmm = self.__get_dataset_non_hmm(self.__corpus);

    def __get_corpus(self):
        with open('id_pud-ud-test.conllu', 'r', encoding='utf8') as fp:

            all_text = fp.read().split('\n\n')

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
        self.__HMM = HMM()
        self.__train()
        self.__test()

        return corpus


    def __get_train_data(self):
        pass

    def __get_test_data(self):
        pass

    def __train(self):
        self.__HMM.train(self.__dataset_basic[:900])
        pass

    def __test(self):
        self.__HMM.test(self.__dataset_basic[901:])
        pass

    def __get_dataset_non_hmm(self, corpus):
        feature_array = []

        for text in corpus:

            for index, word_data in enumerate(text):

                if index == 0:
                    pass

                elif index == len(text)-1:
                    pass

                else:
                    pass

                print(index+word_data[0])



        print(corpus)