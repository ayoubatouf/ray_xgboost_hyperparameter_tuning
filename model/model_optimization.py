from sklearn.model_selection import StratifiedKFold
from data_loader.data_loader import DataLoader
from evaluator.evaluator import Evaluator
from hyperparameter_tuning.hyperparameter_tuner import HyperparameterTuner
from model.model_manager import ModelManager
import xgboost as xgb


class ModelOptimization:
    def __init__(
        self,
        dataset,
        target_column,
        tuner: HyperparameterTuner,
        evaluator: Evaluator,
        model_manager: ModelManager,
    ):
        try:

            self.data_loader = DataLoader()
            self.model_manager = model_manager
            self.tuner = tuner
            self.evaluator = evaluator
            self.X, self.y = self.data_loader.load(dataset, target_column)
        except Exception as e:
            raise RuntimeError(f"Error initializing ModelOptimization: {e}")

    def optimize_and_train(self):
        try:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            feature_names = self.X.columns

            best_params, _ = self.tuner.tune()

            best_model = None
            best_score = 0
            best_metrics = None

            for train_idx, valid_idx in kf.split(self.X, self.y):
                X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_train, y_valid = self.y[train_idx], self.y[valid_idx]

                try:

                    model = xgb.XGBClassifier(**best_params)
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)],
                        verbose=False,
                    )
                except Exception as e:
                    print(f"Error training model in fold: {e}")
                    continue

                try:

                    accuracy, precision, recall, f1, roc_auc = self.evaluator.evaluate(
                        model, X_valid, y_valid
                    )
                except Exception as e:
                    print(f"Error evaluating model in fold: {e}")
                    continue

                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_metrics = (accuracy, precision, recall, f1, roc_auc)

            if best_model is None:
                raise RuntimeError("No valid model found during training.")

            print(f"\nBest Model Metrics:")
            print(f"Accuracy: {best_metrics[0]:.4f}")
            print(f"Precision: {best_metrics[1]:.4f}")
            print(f"Recall: {best_metrics[2]:.4f}")
            print(f"F1 Score: {best_metrics[3]:.4f}")
            print(f"ROC-AUC: {best_metrics[4]:.4f}")

            try:
                self.model_manager.save_model(best_model)
            except Exception as e:
                raise RuntimeError(f"Error saving the model: {e}")

            print("\nFeature Importances:")
            for feature_name, importance in zip(
                feature_names, best_model.feature_importances_
            ):
                print(f"{feature_name}: {importance:.4f}")

        except Exception as e:
            raise RuntimeError(f"An error occurred during optimization: {e}")
