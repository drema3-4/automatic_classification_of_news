def __prepare_data__(self):
    self.data['text'] = self.data[self.columns].apply(
        lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    unique_topics = self.data['topic'].unique()
    self.topic2id = {topic: i for i, topic in enumerate(unique_topics)}
    self.id2topic = {i: topic for i, topic in enumerate(unique_topics)}
    self.num_labels = len(self.topic2id)
    self.data['label'] = self.data['topic'].map(self.topic2id)