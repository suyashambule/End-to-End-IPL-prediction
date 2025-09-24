import os
import sys
from pathlib import Path
from src.IPL_Predictor.config.configuration import ConfigurationManager
from src.IPL_Predictor.components.model_trainer import ModelTrainer
from src.IPL_Predictor import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()

            # Initialize and train model
            model_trainer = ModelTrainer(config=model_trainer_config)
            model = model_trainer.train()
            
            # Log completion
            logger.info(f"Model training completed successfully")
            logger.info(f"Model saved at: {os.path.join(model_trainer_config.root_dir, model_trainer_config.model_name)}")
            
            return model
            
        except Exception as e:
            logger.exception("Error in model training pipeline")
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Model Training Pipeline Started <<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>> Model Training Pipeline Completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e