from src.IPL_Predictor.config.configuration import ConfigurationManager
from src.IPL_Predictor.components.data_validation import DataValidation
from src.IPL_Predictor import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.validate_all_columns()
            
            if not validation_status:
                raise ValueError("Data validation failed. Check validation status file for details.")
                
            return validation_status
            
        except Exception as e:
            logger.exception(e)
            raise e