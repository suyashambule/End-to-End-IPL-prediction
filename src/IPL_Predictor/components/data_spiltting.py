from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.IPL_Predictor.entity.config_entity import DataSplitterConfig
from src.IPL_Predictor import logger
import pandas as pd
import os
import numpy as np

class Data_Splitting:
    def __init__(self, config: DataSplitterConfig, schema=None):
        self.config = config
        self.schema = schema
        if schema is None:
            raise ValueError("Schema is required for data splitting")

    def split_data(self):
        try:
            # Read the data
            df = pd.read_csv(self.config.unzip_data_dir)
            logger.info(f"Read data with shape: {df.shape}")
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.schema.Numerical_COLUMNS),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), 
                     self.schema.Categorial_COLUMNS)
                ]
            )
            
            # Split data
            X = df.drop(self.schema.TARGET_COLUMN, axis=1)
            y = df[self.schema.TARGET_COLUMN]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.split_ratio,
                random_state=self.config.random_state,
                stratify=y
            )
            
            # Fit and transform training data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Get feature names
            num_features = self.schema.Numerical_COLUMNS
            cat_features = []
            for i, col in enumerate(self.schema.Categorial_COLUMNS):
                categories = preprocessor.named_transformers_['cat'].categories_[i][1:]
                cat_features.extend([f"{col}_{cat}" for cat in categories])
            
            # Create processed DataFrames
            train_df = pd.DataFrame(
                X_train_processed,
                columns=num_features + cat_features
            )
            test_df = pd.DataFrame(
                X_test_processed,
                columns=num_features + cat_features
            )
            
            # Add target back
            train_df[self.schema.TARGET_COLUMN] = y_train.values
            test_df[self.schema.TARGET_COLUMN] = y_test.values
            
            # Create output directories and save
            os.makedirs(os.path.dirname(self.config.train_data), exist_ok=True)
            train_df.to_csv(self.config.train_data, index=False)
            test_df.to_csv(self.config.test_data, index=False)
            
            logger.info(f"Processed data saved to {self.config.root_dir}")
            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error in data splitting: {str(e)}")
            raise e