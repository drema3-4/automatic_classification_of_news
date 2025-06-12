def add_regularizer(self, name: str, tau: float = 0.0) -> None:
    if name == "SmoothSparseThetaRegularizer":
        self.model.regularizers.add(
            artm.SmoothSparseThetaRegularizer(name=name, tau=tau)
        )
        self.user_regularizers[name] = tau
    elif name == "SmoothSparsePhiRegularizer":
        self.model.regularizers.add(
            artm.SmoothSparsePhiRegularizer(name=name, tau=tau)
        )
    # остальные регул€ризаторы ...
    else:
        print(
            "–егул€ризатора {0} нет! ѕроверьте корректность названи€!".
            format(name)
        )