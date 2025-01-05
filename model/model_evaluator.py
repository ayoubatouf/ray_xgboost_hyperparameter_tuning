from evaluator.evaluator import Evaluator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import numpy as np


from evaluator.evaluator import Evaluator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import numpy as np


class ModelEvaluator(Evaluator):
    def evaluate(self, model, X_valid, y_valid):
        try:

            if not hasattr(model, "predict"):
                raise AttributeError("The model does not have a 'predict' method.")

            y_pred = model.predict(X_valid)

            accuracy = accuracy_score(y_valid, y_pred)
            precision = precision_score(
                y_valid, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(y_valid, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_valid, y_pred, average="macro", zero_division=0)

            y_valid_bin = label_binarize(y_valid, classes=np.unique(y_valid))

            y_pred_proba = (
                model.predict_proba(X_valid)
                if hasattr(model, "predict_proba")
                else None
            )

            roc_auc = None
            if y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(
                        y_valid_bin, y_pred_proba, multi_class="ovr"
                    )
                except ValueError as e:

                    print(f"Error calculating ROC AUC: {e}")

            return accuracy, precision, recall, f1, roc_auc

        except Exception as e:

            raise RuntimeError(f"An error occurred during model evaluation: {e}")
