def __calc_num_docs_for_words__(self) -> None:
    self.num_docs_for_words = dict()
    for row in range(self.p_data.shape[0]):
        for column in self.processing_columns:
            words = self.p_data.loc[row, column].split(" ")
            for word in words:
                if word in self.num_docs_for_words.keys():
                    self.num_docs_for_words[word] += 1
                else:
                    self.num_docs_for_words[word] = 1
def __calc_up_and_down_threshold__(self):
    self.up_threshold = self.p_data.shape[0] * (
        len(self.processing_columns) / 2.0)
    self.down_threshold = self.p_data.shape[0] / 1000.0
for word in words:
    if self.num_docs_for_words[word] >= self.down_threshold and \
    self.num_docs_for_words[word] <= self.up_threshold:
        new_words.append(word)