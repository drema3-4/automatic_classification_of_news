class Text_preparer:
    def __init__(self, additional_stop_words_path: str = ""):
        '''Инициализация.\n
        additional_stop_words: пользовательский список стоп-слов.'''
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_ru = spacy.load("ru_core_news_sm")

        self.tfidf_corpus = None
        self.tfidf_dictionary = None

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

    def __count_letters_in_token__(self, token: str) -> int:
        num_letters = 0

        for symb in token:
            if ("a" <= symb and symb <= "z") or ("A" <= symb and symb <= "Z"):
                num_letters += 1
            if ("а" <= symb and symb <= "я") or ("А" <= symb and symb <= "Я"):
                num_letters += 1

        return num_letters

    def __strip_non_letters__(self, text: str) -> str:
        return re.sub(r"^[^a-zA-Zа-яА-ЯёЁ]+|[^a-zA-Zа-яА-ЯёЁ]+$", "", text)

    def __processing_token__(self, token: str) -> str:
        new_token = self.__strip_non_letters__(token)

        return new_token if (self.__count_letters_in_token__(new_token) +
                             1.0) / (len(new_token) + 1.0) >= 0.5 else ""

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
                    self.__processing_token__(token.lemma_)
                    for token in self.nlp_en(
                        self.__remove_extra_spaces_and_line_breaks__(part[1])
                    ) if not (token.is_stop) and not (token.is_punct) and
                    len(self.__processing_token__(token.lemma_)) > 1
                ]
            else:
                tokens += [
                    self.__processing_token__(token.lemma_)
                    for token in self.nlp_ru(
                        self.__remove_extra_spaces_and_line_breaks__(part[1])
                    ) if not (token.is_stop) and not (token.is_punct) and
                    len(self.__processing_token__(token.lemma_)) > 1
                ]

        return " ".join(tokens)

    def __calc_tfidf_corpus_without_zero_score_tokens_and_tfidf_dictionary__(
        self
    ) -> None:
        '''Вычисление tfidf метрики для слов документов + tfidf словаря.'''
        texts = []
        self.original_tokens = []

        for row in range(self.p_data.shape[0]):
            words = []
            for column in self.processing_columns:
                for word in self.p_data.loc[row, column].split(" "):
                    words.append(word)
            self.original_tokens.append(words)
            texts.append(words)

        dictionary = gensim.corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = gensim.models.TfidfModel(corpus)

        self.tfidf_corpus = tfidf[corpus]
        self.tfidf_dictionary = dictionary

    def __add_in_tfidf_corpus_zero_score_tokens__(self) -> None:
        '''Добавление слов в tfidf_corpus, которые были исключены gensim при
        подсчёте метрики tfidf (gensim не добавляет слова, которые встречаются
        во всех документах или которые имеют 0 метрику tfidf в tfidf_corpus).'''
        full_corpus = []

        for doc_idx, doc in enumerate(self.tfidf_corpus):
            original_words = self.original_tokens[doc_idx]
            term_weights = {
                self.tfidf_dictionary.get(term_id): weight
                for term_id, weight in doc
            }

            full_doc = []
            for word in original_words:
                if word in term_weights:
                    weight = term_weights[word]
                else:
                    weight = 0.0
                full_doc.append((word, weight))

            full_corpus.append(full_doc)

        self.tfidf_corpus = full_corpus

    def __calc_threshold_for_tfidf_stop_words__(
        self, tfidf_percent_treshold
    ) -> None:
        '''Вычисляет порог tfidf метрики, при котором слова, значение tfidf
        которых меньше, считаются стоп-словами.\n
        tfidf_percent_threshold: процент от всех слов, которые будут считаться
        стоп-словами. То есть берём списко всех значений tfidf, сортируем их и
        значение, которое отсекает от остальной базы 1 процент самых низких
        значений и будет threshold.'''
        all_tfidf_values = []
        for doc in self.tfidf_corpus:
            for _, tfidf_value in doc:
                all_tfidf_values.append(tfidf_value)

        self.threshold_for_tfidf_stop_words = np.percentile(
            all_tfidf_values, tfidf_percent_treshold
        )

    def del_tfidf_stop_words(self, tfidf_percent_treshold) -> None:
        '''Удаляет стоп-слова на основе посчитанного tfidf_corpus и
        tfidf_threshold.'''
        self.__calc_tfidf_corpus_without_zero_score_tokens_and_tfidf_dictionary__(
        )
        self.__add_in_tfidf_corpus_zero_score_tokens__()
        self.__calc_threshold_for_tfidf_stop_words__(tfidf_percent_treshold)

        for row, doc in zip(range(self.p_data.shape[0]), self.tfidf_corpus):
            tfidf_stop_words = [
                word for word, tfidf_value in doc
                if tfidf_value < self.threshold_for_tfidf_stop_words
            ]

            for column in self.processing_columns:
                words_without_tfidf_stop_words = []
                for word in self.p_data.loc[row, column].split(" "):
                    if word in tfidf_stop_words:
                        continue
                    words_without_tfidf_stop_words.append(word)
                self.p_data.loc[
                    row, column] = " ".join(words_without_tfidf_stop_words)

    def __calc_num_docs_for_words__(self) -> None:
        self.num_docs_for_words = dict()

        for row in range(self.p_data.shape[0]):
            for column in self.processing_columns:
                words = self.p_data.loc[row, column].split(" ")

                for word in words:
                    if word in self.num_docs_for_words.keys():
                        self.num_docs_for_words[word] += 1
                    else:
                        self.num_docs_for_words[word] = 1

    def __count_num_words__(self, doc: str) -> int:
        return len(doc.split(" "))

    def __del_docs_with_low_num_words__(self) -> None:
        mask = self.p_data[self.processing_columns].apply(
            lambda col: col.apply(self.__count_num_words__)
        ).sum(axis=1)

        self.p_data = self.p_data[mask >= 80]

        self.p_data = self.p_data.reset_index(drop=True)

    def __calc_up_and_down_threshold__(self):
        self.up_threshold = self.p_data.shape[0] * (
            len(self.processing_columns) / 2.0
        )
        self.down_threshold = self.p_data.shape[0] / 1000.0

    def processing_data(
        self,
        standart_processing: bool = True,
        tfidf_processing: bool = False,
        tfidf_percent_treshold: int = 1
    ) -> None:
        '''
        Функция, вызывающая вышеописанные функции для обработки pandas DataFrame.\n
        standart_processing: bool - говорит нужно ли делать стандартную обработку;\n
        tfidf_processing: bool - говорит нужно ли удалять стоп-слова на основе
        tfidf;\n
        tfidf_percent_treshold: int - какой процент минимальных значений tfidf
        отсеивать.'''
        self.p_data = self.data.copy(deep=True)
        self.p_data.fillna("", inplace=True)

        if standart_processing:
            for row in range(self.p_data.shape[0]):
                for column in self.processing_columns:
                    cell = self.p_data.loc[row, column]

                    if len(cell) > 0:
                        self.p_data.loc[
                            row, column] = self.__processing_cell__(cell)

            self.__del_docs_with_low_num_words__()
            self.__calc_up_and_down_threshold__()
            self.__calc_num_docs_for_words__()

            for row in range(self.p_data.shape[0]):
                for column in self.processing_columns:
                    words = self.p_data.loc[row, column].split(" ")
                    new_words = []

                    for word in words:
                        if self.num_docs_for_words[
                            word
                        ] >= self.down_threshold and self.num_docs_for_words[
                            word] <= self.up_threshold:
                            new_words.append(word)

                    self.p_data.loc[row, column] = " ".join(new_words)

            self.__del_docs_with_low_num_words__()

        if tfidf_processing:
            self.p_data = self.data
            self.p_data = self.p_data.fillna("")
            self.p_data = self.p_data.astype(str)
            self.del_tfidf_stop_words(tfidf_percent_treshold)
            self.__del_docs_with_low_num_words__()

    def add_data(self, data: pd.DataFrame) -> None:
        self.data = data.copy(deep=True)

    def get_data(self) -> pd.DataFrame:
        return self.data

    def add_processing_columns(self, processing_columns: list[str]) -> None:
        self.processing_columns = processing_columns

    def get_processing_columns(self) -> list[str]:
        return self.processing_columns

    def get_processing_data(self) -> pd.DataFrame:
        return self.p_data.copy(deep=True)

    def save_processing_data(self, path: str) -> None:
        self.p_data.to_excel(path, index=False)