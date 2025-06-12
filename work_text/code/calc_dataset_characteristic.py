class Calc_docs_characteristics:
    def __init__(
        self,
        docs: pd.DataFrame,
        columns: list[str],
        num_tokens_column: str = "num_tokens_column",
        tokens_column: str = "tokens_column",
        num_unique_tokens_column: str = "num_unique_tokens_column",
        is_prepearing_docs: bool = False
    ):
        '''Инициализация.\n
    docs: pd.DataFrame - документы, по будут вычислять метрики;\n
    columns: list[str] - столбцы документов, по которым будут вычисляться
    метрики;\n
    num_tokens_column: str - названия столбца, который будет хранить число
    слов в документе;\n
    tokens_column: str - название столбца, который будет хранить слова
    документа;\n
    num_unique_tokens_column: str - названия столбца, который будет хранить
    число уникальных слов документа;\n
    is_prepearing_docs: bool - говорит нужно ли подготовливать данные или
    нет (если для вычисления характеристик подаются неподготовленные данные);\n
    stop_words_frequency: int - частота встречаемости слова в массиве
    документов которая говорит является ли слово с такой (или ниже) частотой
    встречаемости стоп-словом или нет.'''
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_ru = spacy.load("ru_core_news_sm")

        self.docs = docs.copy(deep=True)
        self.docs = self.docs.fillna("")
        self.docs = self.docs.astype(str)
        self.columns = columns
        self.num_tokens_column = num_tokens_column
        self.tokens_column = tokens_column
        self.num_unique_tokens_column = num_unique_tokens_column

        if is_prepearing_docs:
            self.__light_prepeare_docs__()

        self.__calc__()

    def __remove_extra_spaces_and_line_breaks__(self, text: str) -> str:
        '''Удаляет из строки лишние пробелы и переносы строки.\n
    text: строка.\n
    Возврщает строку, с удалёнными лишними пробелами и переносами строк.
    '''
        processed = ""

        if type(text) != str or len(text) == 0:
            return ""

        flag = True
        for symb in text:
            if flag and (symb == " " or symb == "\n"):
                processed += " "
                flag = False

            if symb != " " and symb != "\n":
                flag = True

            if flag:
                processed += symb

        return processed.strip()

    def __first_is_en__(self, cell: str) -> bool:
        '''Определяет начинается строка с символа русского алфавита или
        английского алфавита.\n
        cell: строка.\n
        Возвращает true, если строка начинается с символа английского алфавита.
        '''
        index_first_en = re.search(r"[a-zA-Z]", cell)
        index_first_ru = re.search(r"[а-яА-Я]", cell)

        return True if index_first_en and (
            not (index_first_ru) or
            index_first_en.start() < index_first_ru.start()
        ) else False

    def __split_into_en_and_ru__(self, cell: str) -> list[(bool, str)]:
        '''Разделяет строку на части, в которых содержатся символы принадлежащие
        только русскому или английскому алфавиту (то есть в строке с русскими
        символами не будет символов английского языка и наоборот, остальные символы
        не удаляются).\т
        cell: строка.\n
        Возврщает массив кортежей
        (True(если начинается с символа английского алфавита), подстрока).
        '''
        parts = []
        is_en = self.__first_is_en__(cell)
        part = ""
        for symb in cell:
            if is_en == (symb in string.ascii_letters) or not (symb.isalpha()):
                part += symb
            else:
                parts.append((is_en, part))
                part = symb
                is_en = not (is_en)

        if part:
            parts.append((is_en, part))

        return parts

    def __processing_token__(self, token: str) -> str:
        '''Обрабатывает 1 терм.\n
    token_lemma: строка.\n
    Возвращает обработанный терм.
    '''
        return self.__remove_extra_spaces_and_line_breaks__(token)

    def __processing_cell__(self, cell: str) -> str:
        '''Полностью обрабатывает 1 ячейку pandas DataFrame. То есть проводит
    токенизацию, лемматизацию, удаление стоп слов и перевод в нижний регистр,
    потом происходит склейка и возвращается обработанная ячейка.\n
    cell: строка - ячейка pandas DataFrame.\n
    Возвращает обработанную строку.
    '''
        parts = self.__split_into_en_and_ru__(cell)

        tokens = []

        for part in parts:
            if part[0]:
                tokens += [
                    self.__processing_token__(token.text)
                    for token in self.nlp_en(part[1])
                ]
            else:
                tokens += [
                    self.__processing_token__(token.text)
                    for token in self.nlp_ru(part[1])
                ]

        return " ".join(tokens)

    def __light_prepeare_docs__(self) -> None:
        '''Функция, применяющая обработку данных.'''
        for row in range(self.docs.shape[0]):
            for column in self.docs.columns:
                self.docs.loc[row, column] = self.__processing_cell__(
                    self.docs.loc[row, column]
                )

    def __calc_num_tokens_in_doc__(self, row: int) -> int:
        '''Вычисляет для документа количество токенов в нём.'''
        return sum(
            [
                len(doc.split(" ")) if len(doc) > 0 else 0
                for doc in self.docs.loc[row, self.columns]
            ]
        )

    def __calc_num_tokens_in_docs__(self) -> None:
        '''Вычисляет для всех документов количество токенов в них.'''
        self.docs[self.num_tokens_column] = [
            self.__calc_num_tokens_in_doc__(row)
            for row in range(self.docs.shape[0])
        ]

    def __calc_tokens_in_doc__(self, row: int) -> dict[str, int]:
        '''Вычисляет токены документа и частоту встречаемости их в документе.'''
        tokens = {}
        for column in self.columns:
            words = self.docs.loc[row, column].split(" ")
            for word in words:
                if len(word) > 0:
                    if word in tokens.keys():
                        tokens[word] += 1
                    else:
                        tokens[word] = 1

        return tokens

    def __calc_tokens_in_docs__(self) -> None:
        '''Применяет функцию выше для всех документов.'''
        self.docs[self.tokens_column] = [
            self.__calc_tokens_in_doc__(row)
            for row in range(self.docs.shape[0])
        ]

    def __calc_num_tokens__(self) -> None:
        '''Вычисляет колько всего токенов во всех документах.'''
        self.num_tokens = int(self.docs[self.num_tokens_column].sum())

    def __calc_all_tokens__(self) -> None:
        '''Вычисляет все токены и частоту их встречаемости во всех документах.'''
        self.all_tokens = {}
        for row in range(self.docs.shape[0]):
            for token, frequency in self.docs.loc[row,
                                                  self.tokens_column].items():
                if token in self.all_tokens.keys():
                    self.all_tokens[token] += frequency
                else:
                    self.all_tokens[token] = frequency

    def __calc_num_unique_tokens_in_doc__(self, row: int) -> int:
        '''Вычисляет количество уникальных токенов для документа.'''
        return len(self.docs.loc[row, self.tokens_column].keys())

    def __calc_num_unique_tokens_in_docs__(self) -> None:
        '''Применяет описанную выше функцию для всех документов.'''
        self.docs[self.num_unique_tokens_column] = [
            self.__calc_num_unique_tokens_in_doc__(row)
            for row in range(self.docs.shape[0])
        ]

    def __calc_num_unique_tokens__(self) -> None:
        '''Вычисляет число уникальных токенов для всех документов.'''
        self.num_unique_tokens = len(self.all_tokens.keys())

    def __calc_unique_tokens__(self) -> None:
        '''Вычисляет список уникальных токенов для всех документов.'''
        self.unique_tokens = set(self.all_tokens.keys())

    def __calc_data_for_zips_law__(self) -> None:
        '''Вычисляет данные для закона Ципфа.'''
        self.zips_law_data = (
            list(range(1,
                       len(self.all_tokens.values()) + 1)),
            sorted(list(self.all_tokens.values()), reverse=True)
        )

    def __calc_data_for_heaps_law__(self) -> None:
        '''Вычисляет данные для закона Хипса.'''
        temp = self.docs[[
            self.num_tokens_column, self.num_unique_tokens_column
        ]].copy(deep=True).sort_values(by=self.num_unique_tokens_column)
        self.heaps_law_data = (
            temp[self.num_tokens_column], temp[self.num_unique_tokens_column]
        )

    def __calc_statistic__(self) -> None:
        '''Функция вычисляющая все статистики по документам.'''
        temp = {
            "num docs":
                self.num_docs,
            "num tokens":
                self.num_tokens,
            "num unique tokens":
                self.num_unique_tokens,
            "min num tokens in doc":
                self.min_num_tokens_in_doc,
            "mode num tokens in doc":
                self.mode_num_tokens_in_doc,
            "median num tokens in doc":
                self.median_num_tokens_in_doc,
            "average num tokens in doc":
                self.average_num_tokens_in_doc,
            "max num tokens in doc":
                self.max_num_tokens_in_doc,
            "min num unique tokens in doc":
                self.min_num_unique_tokens_in_doc,
            "mode num unique tokens in doc":
                self.mode_num_unique_tokens_in_doc,
            "median num unique tokens in doc":
                self.median_num_unique_tokens_in_doc,
            "average num unique tokens in doc":
                self.average_num_unique_tokens_in_doc,
            "max num unique tokens in doc":
                self.max_num_unique_tokens_in_doc
        }

        self.statistic = pd.DataFrame(
            {
                "characteristic": list(temp.keys()),
                "value": list(temp.values())
            }
        )

    def __calc_num_docs__(self) -> None:
        '''Вычисляет количество документов в коллекции.'''
        self.num_docs = self.docs.shape[0]

    def __calc_min_num_tokens_in_doc__(self) -> None:
        '''Вычисляет минимальную длину документа.'''
        self.min_num_tokens_in_doc = int(
            self.docs[self.num_tokens_column].min()
        )

    def __calc_mode_num_tokens_in_doc__(self) -> None:
        '''Вычисляет модальную длину документа.'''
        self.mode_num_tokens_in_doc = int(
            self.docs[self.num_tokens_column].mode()[0]
        )

    def __calc_median_num_tokens_in_doc__(self) -> None:
        '''Вычисляет медианную длину документа.'''
        self.median_num_tokens_in_doc = int(
            self.docs[self.num_tokens_column].median()
        )

    def __calc_average_num_tokens_in_doc__(self) -> None:
        '''Вычисляет среднюю длину длкумента.'''
        self.average_num_tokens_in_doc = int(
            self.docs[self.num_tokens_column].mean()
        )

    def __calc_max_num_tokens_in_doc__(self) -> None:
        '''Вычисляет максимальную длину документов.'''
        self.max_num_tokens_in_doc = int(
            self.docs[self.num_tokens_column].max()
        )

    def __calc_min_num_unique_tokens_in_doc__(self) -> None:
        '''Вычисляет минимальное количество уникальных слов в документе.'''
        self.min_num_unique_tokens_in_doc = int(
            self.docs[self.num_unique_tokens_column].min()
        )

    def __calc_mode_num_unique_tokens_in_doc__(self) -> None:
        '''Вычисляет модальное количество уникальных слов в документе.'''
        self.mode_num_unique_tokens_in_doc = int(
            self.docs[self.num_unique_tokens_column].mode()[0]
        )

    def __calc_median_num_unique_tokens_in_doc__(self) -> None:
        '''Вычисляет медианное количество уникальных слов в документе.'''
        self.median_num_unique_tokens_in_doc = int(
            self.docs[self.num_unique_tokens_column].median()
        )

    def __calc_average_num_unique_tokens_in_doc__(self) -> None:
        '''Вычисляет среднее количество уникальных слов в документе.'''
        self.average_num_unique_tokens_in_doc = int(
            self.docs[self.num_unique_tokens_column].mean()
        )

    def __calc_max_num_unique_tokens_in_doc__(self) -> None:
        '''Вычисляет максимальное количество уникальных слов в документе.'''
        self.max_num_unique_tokens_in_doc = int(
            self.docs[self.num_unique_tokens_column].max()
        )

    def __calc_of_additional_stop_words__(self) -> None:
        '''Дополнительно вычисляет неучтённые стоп слова.'''
        self.additional_stop_words = {
            k
            for k, v in self.all_tokens.items() if v <= self.stop_word_frequency
        }

    def __calc__(self) -> None:
        '''Вызов вычисления всех метрик.'''
        self.__calc_num_tokens_in_docs__()
        self.__calc_tokens_in_docs__()
        self.__calc_num_tokens__()
        self.__calc_all_tokens__()
        self.__calc_num_unique_tokens_in_docs__()
        self.__calc_num_unique_tokens__()
        self.__calc_unique_tokens__()
        self.__calc_data_for_zips_law__()
        self.__calc_data_for_heaps_law__()
        self.__calc_num_docs__()
        self.__calc_min_num_tokens_in_doc__()
        self.__calc_mode_num_tokens_in_doc__()
        self.__calc_median_num_tokens_in_doc__()
        self.__calc_average_num_tokens_in_doc__()
        self.__calc_max_num_tokens_in_doc__()
        self.__calc_min_num_unique_tokens_in_doc__()
        self.__calc_mode_num_unique_tokens_in_doc__()
        self.__calc_median_num_unique_tokens_in_doc__()
        self.__calc_average_num_unique_tokens_in_doc__()
        self.__calc_max_num_unique_tokens_in_doc__()
        self.__calc_statistic__()

    def calc(self) -> None:
        self.__calc__()

    def get_docs(self) -> pd.DataFrame:
        return self.docs.copy(deep=True)

    def get_num_docs(self) -> int:
        return self.num_docs

    def get_num_tokens(self) -> int:
        return self.num_tokens

    def get_num_unique_tokens(self) -> int:
        return self.num_unique_tokens

    def get_min_num_tokens_in_doc(self) -> int:
        return self.min_num_tokens_in_doc

    def get_mode_num_tokens_in_doc(self) -> int:
        return self.mode_num_tokens_in_doc

    def get_mode_num_tokens_in_doc(self) -> int:
        return self.median_num_tokens_in_doc

    def get_average_num_tokens_in_doc(self) -> int:
        return self.average_num_tokens_in_doc

    def get_max_num_tokens_in_doc(self) -> int:
        return self.max_num_tokens_in_doc

    def get_tokens_and_frequency(self) -> dict[str, int]:
        return self.all_tokens

    def get_min_num_unique_tokens_in_doc(self) -> int:
        return self.min_num_unique_tokens_in_doc

    def get_mode_num_unique_tokens_in_doc(self) -> int:
        return self.mode_num_unique_tokens_in_doc

    def get_median_num_unique_tokens_in_doc(self) -> int:
        return self.median_num_unique_tokens_in_doc

    def get_average_num_unique_tokens_in_doc(self) -> int:
        return self.average_num_unique_tokens_in_doc

    def get_max_num_unique_tokens_in_doc(self) -> int:
        return self.max_num_unique_tokens_in_doc

    def get_unique_tokens(self) -> set:
        return self.unique_tokens

    def print_log_zips_law(self) -> None:
        plt.plot(
            self.zips_law_data[0], self.zips_law_data[1], label="Zip's law"
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Rank")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def print_abbreviated_zips_law(
        self, start_rank: int, end_rank: int
    ) -> None:
        plt.plot(
            self.zips_law_data[0][start_rank:end_rank],
            self.zips_law_data[1][start_rank:end_rank],
            label="Zip's law"
        )
        plt.xlabel("Rank")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def print_heaps_law(self) -> None:
        plt.plot(
            self.heaps_law_data[0], self.heaps_law_data[1], label="Heap's law"
        )
        plt.xlabel("Lenght of document")
        plt.ylabel("Num unique tokens")
        plt.legend()
        plt.show()

    def get_statistics(self) -> pd.DataFrame:
        return self.statistic

    def print_statistic(self) -> None:
        print(self.statistic)

    def get_n_most_unpopular_tokens(self, n: int) -> dict[str, int]:
        return {
            k: v
            for k, v in
            sorted(self.all_tokens.items(), key=lambda item: item[1])[:n]
        }

    def print_n_most_unpopular_tokens(self, n: int) -> None:
        print(
            {
                k: v
                for k, v in
                sorted(self.all_tokens.items(), key=lambda item: item[1])[:n]
            }
        )

    def get_n_most_popular_tokens(self, n: int) -> dict[str, int]:
        return {
            k: v
            for k, v in sorted(
                self.all_tokens.items(), key=lambda item: item[1], reverse=True
            )[:n]
        }

    def print_n_most_popular_tokens(self, n: int) -> None:
        print(
            {
                k: v
                for k, v in sorted(
                    self.all_tokens.items(),
                    key=lambda item: item[1],
                    reverse=True
                )[:n]
            }
        )

    def get_words_that_occur_from_start_to_end_times(
        self, start: int, end: int
    ) -> list[str]:
        return sorted(
            [k for k, v in self.all_tokens.items() if start <= v or v <= end]
        )

    def print_words_that_occur_from_start_to_end_times(
        self, start: int, end: int
    ) -> None:
        res = sorted(
            [
                (k, v) for k, v in self.all_tokens.items()
                if (v >= start and v <= end)
            ],
            key=lambda x: x[1]
        )
        print(sum([v for k, v in res]))
        print([k for k, v in res])