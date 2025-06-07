def add_regularizers(self, regularizers: dict[str, float]) -> None:
    for regularizer in regularizers:
        self.add_regularizer(regularizer, regularizers[regularizer])