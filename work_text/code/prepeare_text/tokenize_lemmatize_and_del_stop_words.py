def __processing_cell__(self, cell: str) -> str:
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