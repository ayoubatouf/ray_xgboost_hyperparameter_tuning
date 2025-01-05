import joblib
import os


class ModelManager:
    @staticmethod
    def save_model(model, model_name="best_model.pkl"):
        try:
            model_version = 1

            if os.path.exists(model_name):
                try:
                    existing_model = joblib.load(model_name)
                    model_version = existing_model.get("version", 1) + 1
                except Exception as e:

                    print(
                        f"Error loading the existing model for versioning: {e}. Starting with version 1."
                    )
                    model_version = 1

            model_name_versioned = (
                f"{model_name.replace('.pkl', '')}_v{model_version}.pkl"
            )

            try:
                joblib.dump(model, model_name_versioned, compress=3)
                print(f"Model saved as {model_name_versioned}")
            except PermissionError as pe:
                raise PermissionError(
                    f"Permission denied when trying to save the model: {pe}"
                )
            except Exception as e:
                raise RuntimeError(f"An error occurred while saving the model: {e}")

        except Exception as e:

            raise RuntimeError(f"An unexpected error occurred during model saving: {e}")
