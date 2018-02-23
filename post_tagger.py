from experiments.hmm import HMM


class PostTagger:
    def __init__(self):
        self.__dataset_basic = []
        self.__dataset_hmm = []
        self.__dataset_non_hmm = []
        self.__corpus = self.__get_corpus();

    def __get_corpus(self):
        with open('id_pud-ud-test.conllu', 'r', encoding='utf8') as fp:

            all_text = fp.read().split('\n\n')

        fp.close()

        self.__dataset_basic = []

        for block in all_text:

            row = block.split('\n')

            row_data = []

            for term in row[2:]:
                # print(term)
                term_part = term.split()
                row_data.append([term_part[1], term_part[3]])


            self.__dataset_basic.append(row_data)

        # print(self.__dataset_basic[901:])
        self.__HMM = HMM()
        self.__HMM.train(self.__dataset_basic[:900])
        self.__HMM.test(self.__dataset_basic[901:])

    def __get_train_data(self):
        pass

    def __get_test_data(self):
        pass

    def __train(self):
        pass

    def __test(self):
        pass
