def optimizer(self):
    study = optuna.create_study(
        directions=["minimize", "maximize", "maximize"])
    study.optimize(self.__objective__, n_trials=self.n_trials)
    best_trial = self.__select_best_trial__(study, weights=[1, -1, -1])
    best_params = best_trial[0]
    num_topics = best_params["num_topics"]
    # скрытые остальные параметры ...
    # скрытый фрагмент создания финальной модели
    final_model.calc_model()
    self.model = final_model