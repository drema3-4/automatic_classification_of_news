class TopicClassifier:
    def __init__(
        self,
        data_path: str,
        columns: List[str],
        maximum_sequence_length: int = 200,
        output_dir: str = "./model"
    ):
        try:
            self.data = pd.read_excel(data_path)
        except FileNotFoundError:
            raise ValueError(f"File {data_path} not found!")

        self.model_name = "nikitast/multilang-classifier-roberta"
        self.columns = columns
        self.maximum_sequence_length = maximum_sequence_length
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.topic2id: Dict[str, int] = {}
        self.id2topic: Dict[int, str] = {}
        self.num_labels: int = 0
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.evaluation_results: Dict[str, float] = {}

    def __prepare_data__(self):
        self.data['text'] = self.data[self.columns].apply(
            lambda x: ' '.join(x.dropna().astype(str)), axis=1
        )

        unique_topics = self.data['topic'].unique()
        self.topic2id = {topic: i for i, topic in enumerate(unique_topics)}
        self.id2topic = {i: topic for i, topic in enumerate(unique_topics)}

        self.num_labels = len(self.topic2id)
        if self.num_labels < 2:
            raise ValueError("At least 2 classes required for classification")

        self.data['label'] = self.data['topic'].map(self.topic2id)

    def __load_model__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True
        ).to(self.device)

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

    def __compute_metrics__(self, eval_pred) -> Dict[str, float]:
        accuracy_metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
            "f1_micro": f1_score(labels, predictions, average="micro"),
            "f1_macro": f1_score(labels, predictions, average="macro"),
            "f1_weighted": f1_score(labels, predictions, average="weighted"),
        }

        try:
            if logits.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(labels, logits[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    labels, logits, multi_class="ovo", average="macro"
                )
        except ValueError:
            metrics["roc_auc"] = float("nan")

        return metrics

    def __print_final_metrics__(self):
        if not self.evaluation_results:
            raise ValueError("Model not evaluated yet. Call train_model() first")

        print("\n" + "="*50)
        print("Final Model Evaluation Metrics:")
        print("-"*50)
        for metric, value in self.evaluation_results.items():
            if metric not in ["eval_loss", "epoch"]:
                print(f"{metric.upper():<15}: {value:.4f}")
        print("="*50 + "\n")

    def train_model(self):
        self.__prepare_data__()
        train_df, val_df = train_test_split(
            self.data,
            test_size=0.2,
            random_state=42,
            stratify=self.data['topic']
        )

        self.__load_model__()

        train_dataset = self.__tokenize_data__(train_df)
        val_dataset = self.__tokenize_data__(val_df)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            lr_scheduler_type="linear",
            warmup_steps=100,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='./logs',
            logging_steps=10,
            report_to="none",
            save_total_limit=1
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.__compute_metrics__,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        self.trainer.train()

        self.evaluation_results = self.trainer.evaluate()
        self.__print_final_metrics__()

        # self.model.save_pretrained(self.output_dir)
        # self.tokenizer.save_pretrained(self.output_dir)

        # with open(f"{self.output_dir}/id2topic.json", "w") as f:
        #     json.dump({str(k): v for k, v in self.id2topic.items()}, f)

    def load_trained_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

        with open(f"{model_path}/id2topic.json", "r") as f:
            self.id2topic = {int(k): v for k, v in json.load(f).items()}

    def predict(self, text: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.maximum_sequence_length
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        return self.id2topic[predicted_id]