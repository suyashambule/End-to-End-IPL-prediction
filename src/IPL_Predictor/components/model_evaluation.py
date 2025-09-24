import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import joblib
import json
from src.IPL_Predictor import logger
from src.IPL_Predictor.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self):
        try:
            # Load test data and model
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            # Split features and target
            X_test = test_data.drop(self.config.target_column, axis=1)
            y_test = test_data[self.config.target_column]

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            scores = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }

            # Save metrics locally
            os.makedirs(os.path.dirname(self.config.metric_file_path), exist_ok=True)
            with open(self.config.metric_file_path, "w") as f:
                json.dump(scores, f, indent=4)

            # Log with MLflow
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.config.all_params)
                
                # Log metrics
                for metric_name, metric_value in scores.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log model if using remote server
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model, 
                        "model", 
                        registered_model_name="RandomForestModel"
                    )

            logger.info("Model evaluation completed successfully")
            logger.info(f"Metrics saved to: {self.config.metric_file_path}")
            return scores

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise e