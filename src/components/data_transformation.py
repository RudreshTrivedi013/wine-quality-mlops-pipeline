import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_transformer(self):
        """Builds and returns the preprocessing pipeline."""
        try:
            logging.info("Building preprocessing pipeline")

            numerical_features = [
                "fixed acidity", "volatile acidity", "citric acid",
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            logging.info(f"Numerical features: {numerical_features}")
            return num_pipeline, numerical_features

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading train and test data")
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            target_column = "quality"

            # Split features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            # Get pipeline
            pipeline, numerical_features = self.get_transformer()

            # Fit on train, transform both
            X_train_scaled = pipeline.fit_transform(X_train[numerical_features])
            X_test_scaled  = pipeline.transform(X_test[numerical_features])

            # Combine features + target back into arrays
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr  = np.c_[X_test_scaled,  np.array(y_test)]

            logging.info("Transformation complete — saving preprocessor")

            # Save pipeline object
            save_object(self.config.preprocessor_obj_path, pipeline)

            return train_arr, test_arr, self.config.preprocessor_obj_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

    print(f"Train array shape : {train_arr.shape}")
    print(f"Test array shape  : {test_arr.shape}")
    print(f"Preprocessor saved: {preprocessor_path}")