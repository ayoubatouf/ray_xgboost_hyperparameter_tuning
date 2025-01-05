from model.model_evaluator import ModelEvaluator
from model.model_manager import ModelManager
from model.model_optimization import ModelOptimization
from hyperparameter_tuning.ray_hyperparameter_tuner import RayHyperparameterTuner
from hyperparameter_tuning.search_space import search_space
import pandas as pd


class HyperparameterPipeline:
    def __init__(self, dataset, target_column):

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("The dataset must be a pandas DataFrame.")

        if target_column not in dataset.columns:
            raise ValueError(
                f"The target column '{target_column}' does not exist in the dataset."
            )

        self.dataset = dataset
        self.target_column = target_column

    def run_pipeline(self):
        try:

            X = self.dataset.drop(columns=[self.target_column])
            y = self.dataset[self.target_column]

            if not search_space:
                raise ValueError("Search space is invalid or empty.")

            tuner = RayHyperparameterTuner(X, y, X, y, search_space=search_space)

            evaluator = ModelEvaluator()
            model_manager = ModelManager()
            optimizer = ModelOptimization(
                self.dataset, self.target_column, tuner, evaluator, model_manager
            )

            optimizer.optimize_and_train()

        except ValueError as ve:
            raise ValueError(f"ValueError occurred: {ve}")
        except TypeError as te:
            raise TypeError(f"TypeError occurred: {te}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during pipeline execution: {e}")
