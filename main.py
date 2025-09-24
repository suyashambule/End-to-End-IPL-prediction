from src.IPL_Predictor.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.IPL_Predictor.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.IPL_Predictor.pipeline.data_splitter_pipeline import DataSplitterTrainingPipeline
from src.IPL_Predictor.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.IPL_Predictor.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
from src.IPL_Predictor import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"  # Added Validation Stage
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Splitting Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_splitter = DataSplitterTrainingPipeline()
    data_splitter.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

if __name__ == "__main__":
    logger.info(">>>>> IPL Prediction Pipeline Started <<<<<")
    try:
        pass
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        logger.info(">>>>> IPL Prediction Pipeline Completed <<<<<\n\n")