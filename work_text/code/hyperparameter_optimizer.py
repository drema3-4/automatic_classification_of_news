class Hyperparameter_optimizer:
    def __init__(
        self,
        data: pd.DataFrame,
        n_trials: int = 50,
        num_topics: tuple[str, int, int] = ("num_topics", 6, 8),
        num_document_passes: tuple[str, int,
                                   int] = ("num_document_passes", 3, 7),
        num_collection_passes: tuple[str, int,
                                     int] = ("num_collection_passes", 3, 7),
        regularizers: dict[str, tuple[str, float, float]] = {
            "SmoothSparseThetaRegularizer": ('tau_theta', -2.0, 2.0),
            "SmoothSparsePhiRegularizer": ('tau_phi', -2.0, 2.0)
        },
        class_ids: dict[str, float] = {"@default_class": 1.0}
    ):
        self.data = data.copy(deep=True)
        self.n_trials = n_trials
        self.num_topics = num_topics
        self.num_document_passes = num_document_passes
        self.num_collection_passes = num_collection_passes
        self.regularizers = regularizers
        self.class_ids = class_ids

        self.robast_scaler = RobustScaler()

    def __objective__(self, trial) -> tuple[float, float, float]:
        num_topics = trial.suggest_int(
            self.num_topics[0], self.num_topics[1], self.num_topics[2]
        )
        num_document_passes = trial.suggest_int(
            self.num_document_passes[0], self.num_document_passes[1],
            self.num_document_passes[2]
        )
        num_collection_passes = trial.suggest_int(
            self.num_collection_passes[0], self.num_collection_passes[1],
            self.num_collection_passes[2]
        )
        tau_theta = trial.suggest_float(
            self.regularizers["SmoothSparseThetaRegularizer"][0],
            self.regularizers["SmoothSparseThetaRegularizer"][1],
            self.regularizers["SmoothSparseThetaRegularizer"][2]
        )
        tau_phi = trial.suggest_float(
            self.regularizers["SmoothSparsePhiRegularizer"][0],
            self.regularizers["SmoothSparsePhiRegularizer"][1],
            self.regularizers["SmoothSparsePhiRegularizer"][2]
        )
        regularizers = {
            "SmoothSparseThetaRegularizer": tau_theta,
            "SmoothSparsePhiRegularizer": tau_phi
        }
        class_ids = self.class_ids

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

    def __select_best_trial__(self, study, weights):
        """Выбирает trial с минимальной взвешенной суммой метрик."""
        params_and_metrics = [
            (trial.params, trial.values) for trial in study.best_trials
        ]
        metrics = np.array([item[1] for item in params_and_metrics])

        scaled_metrics = np.zeros_like(metrics)
        for i in range(metrics.shape[1]):
            scaler = RobustScaler()
            scaled_column = scaler.fit_transform(metrics[:, i].reshape(-1, 1)
                                                ).flatten()

            if weights[i] < 0:
                scaled_column = -scaled_column
            scaled_metrics[:, i] = scaled_column

        scaled_params_and_metrics = [
            (item[0], item[1], scaled_metrics[i].tolist())
            for i, item in enumerate(params_and_metrics)
        ]

        return min(scaled_params_and_metrics, key=lambda trial: sum(trial[2]))

    def optimizer(self):
        study = optuna.create_study(
            directions=["minimize", "maximize", "maximize"]
        )

        study.optimize(self.__objective__, n_trials=self.n_trials)

        best_trial = self.__select_best_trial__(study, weights=[1, -1, -1])

        best_params = best_trial[0]

        num_topics = best_params["num_topics"]
        num_document_passes = best_params["num_document_passes"]
        num_collection_passes = best_params["num_collection_passes"]
        tau_theta = best_params["tau_theta"]
        tau_phi = best_params["tau_phi"]

        print("best params:")
        print(f"num topics = {num_topics}; num document passes = {num_document_passes};\nnum collection passes = {num_collection_passes}; tau theta = {tau_theta}; tau phi = {tau_phi}.")

        final_model = My_BigARTM_model(
            data=self.data,
            num_topics=num_topics,
            num_document_passes=num_document_passes,
            num_collection_passes=num_collection_passes,
            regularizers={
                "SmoothSparseThetaRegularizer": tau_theta,
                "SmoothSparsePhiRegularizer": tau_phi
            },
            class_ids={"@default_class": 1.0}
        )
        final_model.calc_model()

        self.model = final_model

    def get_model(self) -> My_BigARTM_model:
        return self.model

    def save_model(self, path_model: str = "./drive/MyDrive/model") -> None:
        self.model.model.dump_artm_model(path_model)

    def save_phi(self, path_phi: str = "./drive/MyDrive/phi.xlsx") -> None:
        self.model.model.get_phi().to_excel(path_phi)

    def save_theta(
        self, path_theta: str = "./drive/MyDrive/theta.xlsx"
    ) -> None:
        self.model.model.get_theta().T.to_excel(path_theta)