class Worker_with_topics:
    def __init__(
        self,
        theta_path: str,
        model_path: str,
        news_path: str,
        news_with_true_distribution_path: str,
        column_with_true_topics: str = "tags",
        columns: list[str] = ["url", "title", "summary", "content"]
    ):
        self.theta = pd.read_excel(theta_path).iloc[:, 1:]
        self.model = artm.load_artm_model(model_path)
        self.num_topics = self.model.num_topics
        self.news = pd.read_excel(news_path)
        self.news = self.news.fillna("")
        self.news = self.news.astype(str)
        self.news_with_true_distribution = pd.read_excel(
            news_with_true_distribution_path
        )
        self.news_with_true_distribution = self.news_with_true_distribution.fillna(
            ""
        )
        self.news_with_true_distribution = self.news_with_true_distribution.astype(
            str
        )
        self.column_with_true_topics = column_with_true_topics
        self.columns = columns

        self.__calc__()

    def __calc_clear_theta__(self) -> None:
        self.clear_theta = self.theta[self.theta.sum(axis=1) != 0.0].copy(
            deep=True
        )

    def __calc_clear_news__(self) -> None:
        self.clear_news = self.news.copy(deep=True)
        self.clear_news.drop(
            self.clear_news.index.difference(self.clear_theta.index),
            inplace=True
        )

    def __calc_clear_news_with_true_distribution__(self) -> None:
        self.clear_news_with_true_distribution = self.news_with_true_distribution[
            self.news_with_true_distribution["url"].isin(
                self.clear_news["url"]
            )].copy(deep=True)

    def __reset_indecies__(self) -> None:
        self.clear_news = self.clear_news[["title", "summary",
                                           "content"]].copy(deep=True)
        self.clear_news = self.clear_news.reset_index(drop=True)

        self.clear_news_with_true_distribution = self.clear_news_with_true_distribution[
            ["title", "summary", "content", "tags"]].copy(deep=True)
        self.clear_news_with_true_distribution[
            "tags"] = self.clear_news_with_true_distribution["tags"].apply(
                lambda x: x.split()[0] if x else ""
            )
        self.clear_news_with_true_distribution = self.clear_news_with_true_distribution.reset_index(
            drop=True
        )

        self.clear_theta = self.clear_theta.reset_index(drop=True)

    def __calc_absolute_theta__(self) -> None:
        self.absolute_theta = self.clear_theta.copy(deep=True)

        for row in range(self.absolute_theta.shape[0]):
            max_in_row = max(self.absolute_theta.loc[row, :])
            for column in self.absolute_theta.columns:
                self.absolute_theta.loc[
                    row, column] = 1.0 if self.absolute_theta.loc[
                        row, column] == max_in_row else 0.0

    def __calc_defuzzification_threshold__(self) -> None:
        self.defuzzification_threshold = min(
            [
                max(self.clear_theta.loc[row, :])
                for row in range(self.clear_theta.shape[0])
            ]
        )

    def __calc_defuzzification_theta__(self) -> None:
        self.defuzzification_theta = self.clear_theta.copy(deep=True)

        for row in range(self.defuzzification_theta.shape[0]):
            for column in self.defuzzification_theta.columns:
                value = self.defuzzification_theta.loc[row, column]

                self.defuzzification_theta.loc[
                    row, column
                ] = value if value >= self.defuzzification_threshold else 0.0

    def __calc_topic_news_from_theta__(
        self, num_of_topic: int, theta: pd.DataFrame
    ) -> set[int]:
        news = set()

        for row in range(theta.shape[0]):
            if theta.iloc[row, num_of_topic] > 0.0:
                news.add(row)

        return news

    def __calc_intersection_of_topics__(
        self, theta: pd.DataFrame
    ) -> pd.DataFrame:
        topics_news = [
            self.__calc_topic_news_from_theta__(num_of_topic, theta)
            for num_of_topic in range(self.num_topics)
        ]

        intersection_of_themes = pd.DataFrame(
            index=theta.columns, columns=theta.columns
        )

        for this_topic in range(self.num_topics):
            for other_topic in range(self.num_topics):
                intersection_of_themes.loc[
                    theta.columns[this_topic],
                    theta.columns[other_topic]] = len(
                        topics_news[this_topic].intersection(
                            topics_news[other_topic]
                        )
                    ) * 1.0 / (len(topics_news[this_topic]) + 0.0000001)

        return intersection_of_themes.astype(float)

    def __calc_topics_news_from_theta__(self, theta: pd.DataFrame) -> list[int]:
        return [
            len(self.__calc_topic_news_from_theta__(num_of_topic, theta))
            for num_of_topic in range(self.num_topics)
        ]

    def __calc_labeled_news__(self) -> None:
        self.labeled_news = self.clear_news.copy(deep=True)

        topic_names = {
            key: value
            for key, value in
            zip(self.absolute_theta.columns, self.absolute_theta.columns)
        }

        self.labeled_news["topic"] = self.absolute_theta.idxmax(axis=1)
        self.labeled_news["topic"] = self.labeled_news["topic"].map(topic_names)

    def __calc_docs_topics_theta__(self) -> None:
        temp = dict()

        for row in range(self.labeled_news.shape[0]):
            topic = self.labeled_news.loc[row, "topic"]

            if topic in temp.keys():
                temp[topic].append(row)
            else:
                temp[topic] = [row]

        self.docs_topics_theta = [temp[key] for key in temp]
        self.docs_topics_theta = sorted(self.docs_topics_theta, key=len)[::-1]

    def __calc_docs_topics_true_distribution__(self) -> None:
        temp = dict()

        for row in range(self.clear_news_with_true_distribution.shape[0]):
            topic = self.clear_news_with_true_distribution.loc[row, "tags"]

            if topic in temp.keys():
                temp[topic].append(row)
            else:
                temp[topic] = [row]

        self.docs_topics_true_distribution = [temp[key] for key in temp]
        self.docs_topics_true_distribution = sorted(
            self.docs_topics_true_distribution, key=len
        )[::-1]

    def __calc_difference_between_distributions__(self) -> None:
        if len(self.docs_topics_theta) > len(
            self.docs_topics_true_distribution
        ):
            for _ in range(
                len(self.docs_topics_theta) -
                len(self.docs_topics_true_distribution)
            ):
                self.docs_topics_true_distribution.append([])
        elif len(self.docs_topics_theta) < len(
            self.docs_topics_true_distribution
        ):
            for _ in range(
                len(self.docs_topics_true_distribution) -
                len(self.docs_topics_theta)
            ):
                self.docs_topics_theta.append([])

        total_elements = 0
        matching_elements = 0

        for sublist1, sublist2 in zip(
            self.docs_topics_theta, self.docs_topics_true_distribution
        ):
            set1 = set(sublist1)
            set2 = set(sublist2)

            common = set1 & set2
            matching_elements += len(common)
            total_elements += len(set1)

        if total_elements == 0:
            return 0.0

        self.difference_between_distributions = 100 - (
            matching_elements / total_elements
        ) * 100

    def __calc__(self) -> None:
        self.__calc_clear_theta__()
        self.__calc_clear_news__()
        self.__calc_clear_news_with_true_distribution__()
        self.__reset_indecies__()
        self.__calc_absolute_theta__()
        self.__calc_defuzzification_threshold__()
        self.__calc_defuzzification_theta__()
        self.__calc_labeled_news__()
        self.__calc_docs_topics_theta__()
        self.__calc_docs_topics_true_distribution__()
        self.__calc_difference_between_distributions__()

    def show_intersection_of_topics(self) -> None:
        intersection_of_themes = self.__calc_intersection_of_topics__(
            self.defuzzification_theta
        )

        sns.heatmap(intersection_of_themes, annot=True)
        plt.show()

    def show_number_of_news_in_topics(self) -> None:
        absolute_news_in_topics = self.__calc_topics_news_from_theta__(
            self.absolute_theta
        )
        news_in_topics = self.__calc_topics_news_from_theta__(
            self.defuzzification_theta
        )
        themes = self.defuzzification_theta.columns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        rects1 = ax1.bar(themes, absolute_news_in_topics, color='skyblue')
        ax1.set_ylabel('Количество новостей')
        ax1.set_title('Абсолютное распределение новостей по темам')
        ax1.tick_params(axis='x', rotation=45)

        rects2 = ax2.bar(themes, news_in_topics, color='salmon')
        ax2.set_ylabel('Количество новостей')
        ax2.set_title('Нормированное распределение новостей по темам')
        ax2.tick_params(axis='x', rotation=45)

        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    '{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom'
                )

        autolabel(rects1, ax1)
        autolabel(rects2, ax2)

        plt.tight_layout()
        plt.show()

    def show_topics_words_cores(self) -> None:
        topics_words_cores = self.model.score_tracker["top_tokens"].last_tokens

        for key in topics_words_cores:
            print(f"{key}: {topics_words_cores[key]}")

    def get_labeled_news(self) -> pd.DataFrame:
        return self.labeled_news.copy(deep=True)

    def save_labeled_news(
        self, labeled_news_path: str = "./labeled_newx.xlsx"
    ) -> None:
        self.labeled_news.to_excel(labeled_news_path)

    def show_difference_between_distributions(self) -> None:
        print(self.difference_between_distributions)