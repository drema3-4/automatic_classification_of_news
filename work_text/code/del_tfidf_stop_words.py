def del_tfidf_stop_words(self, tfidf_percent_treshold) -> None:
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