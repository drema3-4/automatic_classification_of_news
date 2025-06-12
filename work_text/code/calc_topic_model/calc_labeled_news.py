def __calc_labeled_news__(self) -> None:
    self.labeled_news = self.clear_news.copy(deep=True)
    topic_names = {
        key: value
        for key, value in
        zip(self.absolute_theta.columns, self.absolute_theta.columns)
    }
    self.labeled_news["topic"] = self.absolute_theta.idxmax(axis=1)
    self.labeled_news["topic"] = self.labeled_news["topic"].map(topic_names)