class My_BigARTM_model():
    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame(),
        num_topics: int = 1,
        num_document_passes: int = 1,
        class_ids: dict[str, float] = {"@default_class": 1.0},
        num_processors: int = 8,
        path_vw: str = "./vw.txt",
        batch_size: int = 1000,
        dir_batches: str = "./batches",
        num_top_tokens: int = 10,
        regularizers: dict[str, float] = {},
        num_collection_passes: int = 1,
        plateau_perplexity: float = 0.1,
        plateau_coherence: float = 0.1,
        plateau_topics_purity: float = 0.1,
        epsilon: float = 0.0000001
    ):
        self.data = data.copy(deep=True)
        self.num_topics = num_topics
        self.num_document_passes = num_document_passes
        self.class_ids = class_ids
        self.num_processors = num_processors
        self.path_vw = path_vw
        self.batch_size = batch_size
        self.dir_batches = dir_batches
        self.num_top_tokens = num_top_tokens
        self.user_regularizers = regularizers
        self.num_collection_passes = num_collection_passes
        self.epsilon = epsilon

        self.perplexity_by_epoch = []
        self.coherence_by_epoch = []
        self.topic_purities_by_epoch = []

        self.plateau_perplexity = plateau_perplexity
        self.plateau_coherence = plateau_coherence
        self.plateau_topics_purity = plateau_topics_purity

        if data.empty:
            print(
                "Чтобы создать модель добавьте данные, на которых будет строиться модель"
            )
        else:
            self.__make_vowpal_wabbit__()
            self.__make_batches__()
            self.__make_model__()

        if self.user_regularizers:
            self.add_regularizers(self.user_regularizers)

    def __make_vowpal_wabbit__(self) -> None:
        f = open(self.path_vw, "w")

        for row in range(self.data.shape[0]):
            string = ""
            for column in self.data.columns:
                string += str(self.data.loc[row, column]) + " "

            f.write("doc_{0} ".format(row) + string.strip() + "\n")

    def __make_batches__(self) -> None:
        self.batches = artm.BatchVectorizer(
            data_path=self.path_vw,
            data_format="vowpal_wabbit",
            batch_size=self.batch_size,
            target_folder=self.dir_batches
        )

        self.dictionary = self.batches.dictionary

    def __make_model__(self) -> None:
        self.model = artm.ARTM(
            cache_theta=True,
            num_topics=self.num_topics,
            num_document_passes=self.num_document_passes,
            dictionary=self.dictionary,
            class_ids=self.class_ids,
            num_processors=8
        )

        self.__add_BigARTM_metrics__()

    def __add_BigARTM_metrics__(self) -> None:
        self.model.scores.add(
            artm.PerplexityScore(name='perplexity', dictionary=self.dictionary)
        )
        self.model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
        self.model.scores.add(
            artm.SparsityThetaScore(name='sparsity_theta_score')
        )
        self.model.scores.add(
            artm.TopTokensScore(
                name="top_tokens", num_tokens=self.num_top_tokens
            )
        )

    def __calc_coherence__(self) -> None:
        topics = []
        if "top_tokens" in self.model.score_tracker:
            last_tokens = self.model.score_tracker["top_tokens"].last_tokens
            topics = [last_tokens[topic] for topic in last_tokens]

        valid_topics = []
        for topic in topics:
            if isinstance(topic, list) and len(topic) > 0:
                valid_topics.append(topic)

        if not valid_topics:
            self.coherence = 0.0
            return

        texts = []
        for row in range(self.data.shape[0]):
            words = []
            for column in self.data.columns:
                cell_content = self.data.loc[row, column]
                if isinstance(cell_content, str) and cell_content.strip():
                    words += cell_content.split()
            if words:
                texts.append(words)

        if not texts:
            self.coherence = 0.0
            return

        try:
            dictionary = Dictionary(texts)
            coherence_model = CoherenceModel(
                topics=valid_topics,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v"
            )
            self.coherence = coherence_model.get_coherence()
        except Exception as e:
            print(f"Ошибка при расчете когерентности: {e}")
            self.coherence = 0.0

    def __calc_phi__(self) -> None:
        self.phi = np.sort(self.model.get_phi(), axis=0)[::-1, :]

    def __calc_theta__(self) -> None:
        self.theta = self.model.get_theta()

    def __calc_topic_purity__(self, topic: int) -> None:
        return np.sum(self.phi[:, topic]) / self.phi.shape[0]

    def __calc_topics_purities__(self) -> None:
        topics = range(self.phi.shape[1])
        self.topic_purities = sum(
            [self.__calc_topic_purity__(topic) for topic in topics]
        ) / len(topics)

    def __calc_metrics__(self) -> None:
        self.perplexity = self.model.score_tracker['perplexity'].last_value
        self.sparsity_phi_score = self.model.score_tracker['sparsity_phi_score'
                                                          ].last_value
        self.sparsity_theta_score = self.model.score_tracker[
            'sparsity_theta_score'].last_value
        self.top_tokens = self.model.score_tracker['top_tokens'].last_tokens
        self.__calc_coherence__()
        self.__calc_phi__()
        self.__calc_topics_purities__()

    def add_data(self, data: pd.DataFrame) -> None:
        self.data = data

        self.__make_vowpal_wabbit__()
        self.__make_batches__()
        self.__make_model__()

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
            self.user_regularizers[name] = tau
        elif name == "DecorrelatorPhiRegularizer":
            self.model.regularizers.add(
                artm.DecorrelatorPhiRegularizer(name=name, tau=tau)
            )
            self.user_regularizers[name] = tau
        elif name == "LabelRegularizationPhiRegularizer":
            self.model.regularizers.add(
                artm.LabelRegularizationPhiRegularizer(name=name, tau=tau)
            )
            self.user_regularizers[name] = tau
        elif name == "HierarchicalSparsityPhiRegularizer":
            self.model.regularizers.add(
                artm.HierarchicalSparsityPhiRegularizer(name=name, tau=tau)
            )
            self.user_regularizers[name] = tau
        elif name == "TopicSelectionThetaRegularizer":
            self.model.regularizers.add(
                artm.TopicSelectionThetaRegularizer(name=name, tau=tau)
            )
            self.user_regularizers[name] = tau
        elif name == "BitermsPhiRegularizer":
            self.model.regularizers.add(
                artm.BitermsPhiRegularizer(name=name, tau=tau)
            )
            self.user_regularizers[name] = tau
        elif name == "BackgroundTopicsRegularizer":
            self.model.regularizers.add(
                artm.BackgroundTopicsRegularizer(name=name, tau=tau)
            )
            self.user_regularizers[name] = tau
        else:
            print(
                "Регуляризатора {0} нет! Проверьте корректность названия!".
                format(name)
            )

    def add_regularizers(self, regularizers: dict[str, float]) -> None:
        for regularizer in regularizers:
            self.add_regularizer(regularizer, regularizers[regularizer])

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
                change_coherence_by_percent = abs(self.coherence_by_epoch[epoch - 1] - self.coherence_by_epoch[epoch]) / \
                                              (self.coherence_by_epoch[epoch - 1] + self.epsilon) * 100
                change_topics_purity_by_percent = abs(
                    self.topic_purities_by_epoch[epoch - 1] - self.topic_purities_by_epoch[epoch]) / \
                                                  (self.topic_purities_by_epoch[epoch - 1] + self.epsilon) * 100

                if change_perplexity_by_percent < self.plateau_perplexity and change_coherence_by_percent < self.plateau_coherence and change_topics_purity_by_percent < self.plateau_topics_purity:
                    break

    def get_perplexity(self) -> float:
        return self.perplexity

    def get_perplexity_by_epochs(self) -> list[float]:
        return self.perplexity_by_epoch

    def print_perplexity_by_epochs(self) -> None:
        plt.plot(
            range(len(self.perplexity_by_epoch)),
            self.perplexity_by_epoch,
            label="perplexity"
        )
        plt.title("График перплексии")
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.show()

    def get_coherence(self) -> float:
        return self.coherence

    def get_coherence_by_epochs(self) -> list[float]:
        return self.coherence_by_epoch

    def print_coherence_by_epochs(self) -> None:
        plt.plot(
            range(len(self.coherence_by_epoch)),
            self.coherence_by_epoch,
            label="coherence"
        )
        plt.title("График когерентности")
        plt.xlabel("Epoch")
        plt.ylabel("Coherence")
        plt.legend()
        plt.show()

    def get_topic_purities(self) -> float:
        return self.topic_purities

    def get_topic_purities_by_epochs(self) -> list[float]:
        return self.topic_purities_by_epoch

    def print_topic_purities_by_epochs(self) -> None:
        plt.plot(
            range(len(self.topic_purities_by_epoch)),
            self.topic_purities_by_epoch,
            label="topic purities"
        )
        plt.title("График чистоты тем")
        plt.xlabel("Epoch")
        plt.ylabel("Topics purity")
        plt.legend()
        plt.show()

    def get_model(self):
        return self.model

    def save_model(self, dir_model: str = "./drive/MyDrive/model") -> None:
        self.model.dump_artm_model(dir_model)