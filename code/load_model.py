self.model = AutoModelForSequenceClassification.from_pretrained(
    self.model_name,
    num_labels=self.num_labels,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True
).to(self.device)