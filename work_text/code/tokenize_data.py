def __tokenize_data__(self, df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df[['text', 'label']])
    def tokenize_function(examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.maximum_sequence_length
        )
    return dataset.map(tokenize_function, batched=True)