def calc_threshold_for_tfidf_stop_words(self, tfidf_percent_treshold) -> None:
        all_tfidf_values = []
        for doc in self.tfidf_corpus:
            for _, tfidf_value in doc:
                all_tfidf_values.append(tfidf_value)
        self.threshold_for_tfidf_stop_words = np.percentile(all_tfidf_values, tfidf_percent_treshold)
