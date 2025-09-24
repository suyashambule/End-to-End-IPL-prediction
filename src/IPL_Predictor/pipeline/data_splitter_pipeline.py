from src.IPL_Predictor.config.configuration import ConfigurationManager
from src.IPL_Predictor.components.data_spiltting import Data_Splitting
from src.IPL_Predictor import logger

class DataSplitterTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_split_config = config.get_data_split_config()
            
            # Verify schema exists
            if not hasattr(config, 'schema'):
                raise ValueError("Schema not found in configuration")
                
            data_splitter = Data_Splitting(
                config=data_split_config,
                schema=config.schema
            )
            train_data, test_data = data_splitter.split_data()
            
            logger.info(f"Data splitting completed. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            
        except Exception as e:
            logger.exception(e)
            raise e