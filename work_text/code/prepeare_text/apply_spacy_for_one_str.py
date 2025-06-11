self.nlp_ru = spacy.load("ru_core_news_sm")
parts = self.__split_into_en_and_ru__(cell)
if part[1]:
    tokens += [
        self.__processing_token__(token.lemma_)
        for token in self.nlp_ru(
            self.__remove_extra_spaces_and_line_breaks__(part[1])
        ) if not (token.is_stop) and not (token.is_punct) and
        len(self.__processing_token__(token.lemma_)) > 1
    ]