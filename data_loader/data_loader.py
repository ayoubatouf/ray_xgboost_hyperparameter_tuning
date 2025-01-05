import pandas as pd


class DataLoader:
    def load(self, dataset, target_column):

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("The dataset must be a pandas DataFrame.")

        if target_column not in dataset.columns:
            raise ValueError(
                f"The target column '{target_column}' does not exist in the dataset."
            )

        try:
            X = dataset.drop(columns=[target_column])
            y = dataset[target_column].values
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the dataset: {e}")

        return X, y
