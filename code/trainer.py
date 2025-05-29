self.trainer = Trainer(
    model=self.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=self.__compute_metrics__,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
self.trainer.train()