def calc_model(self):
    self.perplexity_by_epoch = []
    self.coherence_by_epoch = []
    self.topic_purities_by_epoch = []

    for epoch in range(self.num_collection_passes):
        self.model.fit_offline(
            batch_vectorizer=self.batches, num_collection_passes=1
        )
        self.__calc_metrics__()
        self.perplexity_by_epoch.append(self.perplexity)
        self.coherence_by_epoch.append(self.coherence)
        self.topic_purities_by_epoch.append(self.topic_purities)

        if epoch > 0:
            change_perplexity_by_percent = abs(
                self.perplexity_by_epoch[epoch - 1] -
                self.perplexity_by_epoch[epoch]
            ) / (self.perplexity_by_epoch[epoch - 1] + self.epsilon) * 100
            change_coherence_by_percent = \
                abs( self.coherence_by_epoch[epoch - 1] - \
                     self.coherence_by_epoch[epoch] ) / \
                    ( self.coherence_by_epoch[epoch - 1] + \
                      self.epsilon ) * 100
            change_topics_purity_by_percent = \
                abs( self.topic_purities_by_epoch[epoch - 1] - \
                     self.topic_purities_by_epoch[epoch]) / \
                    ( self.topic_purities_by_epoch[epoch - 1] + \
                      self.epsilon ) * 100

            if change_perplexity_by_percent < \
               self.plateau_perplexity and \
               change_coherence_by_percent < \
               self.plateau_coherence and \
               change_topics_purity_by_percent < \
               self.plateau_topics_purity:
                break