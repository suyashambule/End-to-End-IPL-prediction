import os
import pandas as pd
from src.IPL_Predictor.entity.config_entity import DataValidationConfig
from src.IPL_Predictor import logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = False
            
            # Read the dataset
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = list(self.config.all_schema.keys())

            # Check for missing or extra columns
            for col in all_schema:
                if col not in all_cols:
                    validation_status = False
                    logger.error(f"Column: {col} is not found in the dataset")
                    return validation_status

            validation_status = True
            logger.info("All columns are validated")

            # Save validation status
            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            logger.error(f"Error occurred during data validation: {str(e)}")
            raise e