def __select_best_trial__(self, study, weights):
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