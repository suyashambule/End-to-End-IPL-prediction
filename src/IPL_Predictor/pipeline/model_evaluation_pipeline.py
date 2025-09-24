import os
import sys
from pathlib import Path
from src.IPL_Predictor.config.configuration import ConfigurationManager
from src.IPL_Predictor.components.model_evaluation import ModelEvaluation
from src.IPL_Predictor import logger

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()

            # Initialize and run model evaluation
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            scores = model_evaluation.evaluate_model()
            
            # Log results
            logger.info("Model Evaluation Results:")
            for metric, value in scores.items():
                logger.info(f"{metric}: {value:.4f}")
            
            logger.info(f"Metrics saved at: {model_evaluation_config.metric_file_path}")
            
            return scores
            
        except Exception as e:
            logger.exception("Error in model evaluation pipeline")
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Model Evaluation Pipeline Started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> Model Evaluation Pipeline Completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e