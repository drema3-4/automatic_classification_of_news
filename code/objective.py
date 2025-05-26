def __objective__(self, trial) -> tuple[float, float, float]:
    num_topics = trial.suggest_int(
        self.num_topics[0], self.num_topics[1], self.num_topics[2]
    )
    # скрытые остальные гиперпараметры ...
    model = My_BigARTM_model(
        data=self.data,
        num_topics=num_topics,
        num_document_passes=num_document_passes,
        class_ids=class_ids,
        num_collection_passes=num_collection_passes,
        regularizers=regularizers
    )
    model.calc_model()
    return model.get_perplexity(), model.get_coherence(
    ), model.get_topic_purities()