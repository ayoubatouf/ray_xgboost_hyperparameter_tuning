from hyperparameter_tuning.hyperparameter_tuner import HyperparameterTuner
from model.model_evaluator import ModelEvaluator
import xgboost as xgb
from ray import tune
import pandas as pd


class RayHyperparameterTuner(HyperparameterTuner):
    def __init__(self, X_train, y_train, X_valid, y_valid, search_space):

        if not isinstance(X_train, (pd.DataFrame, pd.Series)) or not isinstance(
            y_train, (pd.Series, pd.DataFrame)
        ):
            raise TypeError("X_train and y_train must be pandas DataFrame or Series.")
        if not isinstance(X_valid, (pd.DataFrame, pd.Series)) or not isinstance(
            y_valid, (pd.Series, pd.DataFrame)
        ):
            raise TypeError("X_valid and y_valid must be pandas DataFrame or Series.")
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError("The search_space must be a non-empty dictionary.")

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.search_space = search_space

    def tune(self):
        def objective_ray(config):
            try:

                model = xgb.XGBClassifier(**config)
                eval_set = [(self.X_train, self.y_train), (self.X_valid, self.y_valid)]

                model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)

                accuracy, _, _, _, _ = ModelEvaluator().evaluate(
                    model, self.X_valid, self.y_valid
                )

                return {"accuracy": accuracy}
            except Exception as e:

                return {"accuracy": 0.0, "error": str(e)}

        try:

            analysis = tune.run(
                objective_ray,
                config=self.search_space,
                num_samples=5,  # incease iterations for accurate results
                resources_per_trial={"cpu": 1},
                stop={"accuracy": 0.9},
                metric="accuracy",
                mode="max",
            )

            best_trial = analysis.best_trial
            return best_trial.config, best_trial.last_result["accuracy"]

        except Exception as e:
            raise RuntimeError(f"An error occurred during hyperparameter tuning: {e}")
