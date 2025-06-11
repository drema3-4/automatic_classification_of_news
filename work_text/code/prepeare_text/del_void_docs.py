def __count_num_words__(self, doc: str) -> int:
    return len(doc.split(" "))
def __del_docs_with_low_num_words__(self) -> None:
    mask = self.p_data[self.processing_columns].apply(
        lambda col: col.apply(self.__count_num_words__)
    ).sum(axis=1)
    self.p_data = self.p_data[mask >= 80]
    self.p_data = self.p_data.reset_index(drop=True)