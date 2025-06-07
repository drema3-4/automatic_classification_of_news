def __calc_coherence__(self) -> None:
    last_tokens = self.model.score_tracker["top_tokens"].last_tokens
    valid_topics = [tokens for tokens in last_tokens.values() if tokens]
    texts = []
    for row in range(self.data.shape[0]):
        words = []
        for column in self.data.columns:
            cell_content = self.data.loc[row, column]
            if isinstance(cell_content, str) and cell_content.strip():
                words += cell_content.split()
        if words:
            texts.append(words)
    dictionary = Dictionary(texts)
    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"
    )
    self.coherence = coherence_model.get_coherence()