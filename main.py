import pandas as pd
from hyperparameter_tuning.hyperparameter_pipeline import HyperparameterPipeline

if __name__ == "__main__":
    dataset = pd.read_csv("data/updated_pollution_dataset_encoded.csv")
    target_column = "Air Quality"
    hyperparameter_pipeline = HyperparameterPipeline(dataset, target_column)
    hyperparameter_pipeline.run_pipeline()
