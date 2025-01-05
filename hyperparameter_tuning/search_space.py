from ray import tune

search_space = {
    "learning_rate": tune.uniform(0.001, 0.5),
    "max_depth": tune.randint(3, 21),
    "n_estimators": tune.randint(50, 5001),
    "subsample": tune.uniform(0.5, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
    "gamma": tune.uniform(0, 10),
    "reg_alpha": tune.uniform(0, 5),
    "reg_lambda": tune.uniform(0.1, 5),
    "min_child_weight": tune.randint(1, 21),
}
