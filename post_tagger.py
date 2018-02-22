
class PostTagger:

    def __init__(self):
        self.__corpus = self.__get_corpus();

    def __get_corpus(self):
        with open('id_pud-ud-test.conllu', 'r', encoding='utf8') as fp:

            all_text = fp.read().split('\n\n')

        fp.close()

        print(len(all_text))


    def __get_train_data(self):
        pass

    def __get_test_data(self):
        pass

    def __train(self):
        pass

    def __test(self):
        pass