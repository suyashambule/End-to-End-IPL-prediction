from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from src.IPL_Predictor.entity.config_entity import ModelTrainerConfig
from src.IPL_Predictor import logger
import os
from dotenv import load_dotenv
load_dotenv()

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        logger.info(f"Available columns in training data: {train_data.columns.tolist()}")
        logger.info(f"Target column from config: {self.config.target_column}")
        if self.config.target_column not in train_data.columns:
            available_columns = train_data.columns.tolist()
            logger.info(f"Target column '{self.config.target_column}' not found.")
            logger.info(f"Available columns: {available_columns}")
            raise ValueError(f"Target column '{self.config.target_column}' not found in training data. Available columns: {available_columns}")

        X_train = train_data.drop(self.config.target_column, axis=1)
        y_train = train_data[self.config.target_column]
        X_test = test_data.drop(self.config.target_column, axis=1)
        y_test = test_data[self.config.target_column]
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        rf.fit(X_train, y_train)
        

        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))
        
        return rf