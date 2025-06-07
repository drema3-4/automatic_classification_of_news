result = " ".join(
    [
        token.lemma_
        for token in self.nlp_en(self.__processing_token__(russian_str)) if
        not (token.is_stop) and not (token.is_punct) and len(token.lemma_) > 1
    ]
)