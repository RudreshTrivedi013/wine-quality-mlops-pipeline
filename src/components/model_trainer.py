import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}
            for name, model in models.items():
                logging.info(f"Training: {name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2    = r2_score(y_test, y_pred)
                mae   = mean_absolute_error(y_test, y_pred)
                rmse  = np.sqrt(mean_squared_error(y_test, y_pred))

                report[name] = {
                    "model":  model,
                    "R2":     round(r2,   4),
                    "MAE":    round(mae,  4),
                    "RMSE":   round(rmse, 4),
                }
                logging.info(f"{name} → R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting train/test arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree":     DecisionTreeRegressor(random_state=42),
                "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "XGBoost":           XGBRegressor(
                                     n_estimators=200,
                                     learning_rate=0.05,
                                     max_depth=5,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     random_state=42,
                                     verbosity=0
                                     ),
                }

            logging.info("Starting model evaluation")
            report = self.evaluate_models(X_train, y_train, X_test, y_test, models)

            # Print comparison table
            print("\n" + "="*60)
            print(f"{'Model':<25} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
            print("="*60)
            for name, metrics in report.items():
                print(f"{name:<25} {metrics['R2']:>8} {metrics['MAE']:>8} {metrics['RMSE']:>8}")
            print("="*60)

            # Pick best model by R2
            best_name = max(report, key=lambda x: report[x]["R2"])
            best_model = report[best_name]["model"]
            best_r2    = report[best_name]["R2"]

            print(f"\n✅ Best Model : {best_name}")
            print(f"   R2 Score   : {best_r2}")

            if best_r2 < 0.5:
                raise CustomException("No model achieved R2 >= 0.5", sys)

            # Save best model
            save_object(self.config.trained_model_path, best_model)
            logging.info(f"Best model '{best_name}' saved to {self.config.trained_model_path}")

            return best_name, best_r2, self.config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Transformation
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    # Training
    trainer = ModelTrainer()
    best_name, best_r2, model_path = trainer.initiate_model_training(train_arr, test_arr)

    print(f"\nModel saved at: {model_path}")