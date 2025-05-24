class Text_preparer:
    def __init__( self,
                  additional_stop_words_path: str = ""):
        '''�������������.\n
        additional_stop_words: ���������������� ������ ����-����.'''
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_ru = spacy.load("ru_core_news_sm")
        self.additional_stop_words = set()

        self.tfidf_corpus = None
        self.tfidf_dictionary = None


    def __first_is_en__(self, cell: str) -> bool:
        '''���������� ���������� ������ � ������� �������� �������� ���
        ����������� ��������.\n
        cell: ������.\n
        ���������� true, ���� ������ ���������� � ������� ����������� ��������.
        '''

        index_first_en = re.search(r"[a-zA-Z]", cell)
        index_first_ru = re.search(r"[�-��-�]", cell)

        return True if index_first_en and (not(index_first_ru) or index_first_en.start() < index_first_ru.start()) else False

    def __split_into_en_and_ru__(self, cell: str) -> list[(bool, str)]:
        '''��������� ������ �� �����, � ������� ���������� ������� �������������
        ������ �������� ��� ����������� �������� (�� ���� � ������ � ��������
        ��������� �� ����� �������� ����������� ����� � ��������, ��������� �������
        �� ���������).\�
        cell: ������.\n
        ��������� ������ ��������
        (True(���� ���������� � ������� ����������� ��������), ���������).
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
        '''������� �� ������ ������ ������� � �������� ������.\n
        text: ������.\n
        ��������� ������, � ��������� ������� ��������� � ���������� �����.
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

    def __time_processing__(self, text: str) -> str:
        '''������������ ������, ���������� ����.\n
        text: ������, ���������� ����� � ������� dd:dd.\n
        ���������� ������ ������� ������ time.
        '''

        if re.match(r"\d{2}:\d{2}", text):
            return "time"
        else:
            return text.replace(":", "")

    def __processing_token__(self, token_lemma: str) -> str:
        '''������������ 1 ����.\n
        token_lemma: ������.\n
        ���������� ������������ ����.
        '''

        return self.__time_processing__(
            self.__remove_extra_spaces_and_line_breaks__(token_lemma)
        )

    def __processing_cell__(self, cell: str) -> str:
        '''��������� ������������ 1 ������ pandas DataFrame. �� ���� ��������
        �����������, ������������, �������� ���� ���� � ������� � ������ �������,
        ����� ���������� ������� � ������������ ������������ ������.\n
        cell: ������ - ������ pandas DataFrame.\n
        ���������� ������������ ������.
        '''
        parts = self.__split_into_en_and_ru__(cell)

        tokens = []

        for part in parts:
            if part[0]:
                tokens += [
                    token.lemma_
                    for token in self.nlp_en(self.__processing_token__(part[1]))
                    if not (token.is_stop) and not (token.is_punct) and
                    len(token.lemma_) > 1
                ]
            else:
                tokens += [
                    token.lemma_
                    for token in self.nlp_ru(self.__processing_token__(part[1]))
                    if not (token.is_stop) and not (token.is_punct) and
                    len(token.lemma_) > 1
                ]
                
        return " ".join(tokens)

    def calc_tfidf_corpus_without_zero_score_tokens_and_tfidf_dictionary(self) -> None:
        '''���������� tfidf ������� ��� ���� ���������� + tfidf �������.'''
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

    def add_in_tfidf_corpus_zero_score_tokens(self) -> None:
        '''���������� ���� � tfidf_corpus, ������� ���� ��������� gensim ���
        �������� ������� tfidf (gensim �� ��������� �����, ������� �����������
        �� ���� ���������� ��� ������� ����� 0 ������� tfidf � tfidf_corpus).'''
        full_corpus = []

        for doc_idx, doc in enumerate(self.tfidf_corpus):
            original_words = self.original_tokens[doc_idx]
            term_weights = {self.tfidf_dictionary.get(term_id): weight for term_id, weight in doc}

            full_doc = []
            for word in original_words:
                if word in term_weights:
                    weight = term_weights[word]
                else:
                    weight = 0.0
                full_doc.append((word, weight))

            full_corpus.append(full_doc)

        self.tfidf_corpus = full_corpus

    def calc_threshold_for_tfidf_stop_words(self, tfidf_percent_treshold) -> None:
        '''��������� ����� tfidf �������, ��� ������� �����, �������� tfidf
        ������� ������, ��������� ����-�������.\n
        tfidf_percent_threshold: ������� �� ���� ����, ������� ����� ���������
        ����-�������. �� ���� ���� ������ ���� �������� tfidf, ��������� �� �
        ��������, ������� �������� �� ��������� ���� 1 ������� ����� ������
        �������� � ����� threshold.'''
        all_tfidf_values = []
        for doc in self.tfidf_corpus:
            for _, tfidf_value in doc:
                all_tfidf_values.append(tfidf_value)

        self.threshold_for_tfidf_stop_words = np.percentile(all_tfidf_values, tfidf_percent_treshold)

    def del_tfidf_stop_words(self, tfidf_percent_treshold) -> None:
        '''������� ����-����� �� ������ ������������ tfidf_corpus �
        tfidf_threshold.'''
        self.calc_tfidf_corpus_without_zero_score_tokens_and_tfidf_dictionary()
        self.add_in_tfidf_corpus_zero_score_tokens()
        self.calc_threshold_for_tfidf_stop_words(tfidf_percent_treshold)

        for row, doc in zip(range(self.p_data.shape[0]), self.tfidf_corpus):
            tfidf_stop_words = [word for word, tfidf_value in doc if tfidf_value < self.threshold_for_tfidf_stop_words]

            for column in self.processing_columns:
                words_without_tfidf_stop_words = []
                for word in self.p_data.loc[row, column].split(" "):
                    if word in tfidf_stop_words:
                        continue
                    words_without_tfidf_stop_words.append(word)
                self.p_data.loc[row, column] = " ".join(words_without_tfidf_stop_words)

    def processing_data(self, standart_processing: bool = True, tfidf_processing: bool = False, tfidf_percent_treshold: int = 1) -> None:
        '''
        �������, ���������� ������������� ������� ��� ��������� pandas DataFrame.\n
        standart_processing: bool - ������� ����� �� ������ ����������� ���������;\n
        tfidf_processing: bool - ������� ����� �� ������� ����-����� �� ������
        tfidf;\n
        tfidf_percent_treshold: int - ����� ������� ����������� �������� tfidf
        ���������.'''
        self.p_data = self.data.copy(deep=True)
        self.p_data.fillna("", inplace=True)

        if standart_processing:
            for column in self.processing_columns:
                for row in range(self.p_data.shape[0]):
                    cell = self.p_data.loc[row, column]

                    if type(cell) == str and len(cell) > 0:
                        self.p_data.loc[row, column] = self.__processing_cell__(
                            self.p_data.loc[row, column]
                        )
                    else:
                        self.p_data.loc[row, column] = ""

        if tfidf_processing:
            self.p_data = self.data
            self.p_data = self.p_data.fillna("")
            self.p_data = self.p_data.astype(str)
            self.del_tfidf_stop_words(tfidf_percent_treshold)

    def add_data(self, data: pd.DataFrame) -> None:
        self.data = data.copy(deep=True)

    def get_data(self) -> pd.DataFrame:
        return self.data

    def add_additional_stop_words(self, additional_stop_words_path: str) -> None:
        with open(additional_stop_words_path, "r", encoding="utf-8") as file:
            for stop_word in set(file.read().strip(" ")):
                self.additional_stop_words.add(stop_word)

    def get_additional_stop_words(self) -> set[str]:
        return self.additional_stop_words

    def add_processing_columns(self, processing_columns: list[str]) -> None:
        self.processing_columns = processing_columns

    def get_processing_columns(self) -> list[str]:
        return self.processing_columns

    def get_processing_data(self) -> pd.DataFrame:
        return self.p_data.copy(deep=True)

    def save_processing_data(self, path: str) -> None:
        self.p_data.to_excel(path)