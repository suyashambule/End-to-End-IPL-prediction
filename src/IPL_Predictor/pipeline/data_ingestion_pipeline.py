import os
from src.IPL_Predictor.config.configuration import ConfigurationManager
from src.IPL_Predictor.components.data_ingestion import DataIngestion
from src.IPL_Predictor import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            
            logger.info("Data ingestion completed successfully")
            
        except Exception as e:
            logger.exception(e)
            raise e