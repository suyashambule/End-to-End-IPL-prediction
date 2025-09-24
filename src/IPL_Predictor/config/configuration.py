from pathlib import Path
from src.IPL_Predictor.constants import *
from src.IPL_Predictor.utils.common import read_yaml, create_directories
from pathlib import Path
from src.IPL_Predictor.constants import *
from src.IPL_Predictor.utils.common import read_yaml, create_directories
from src.IPL_Predictor.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataSplitterConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=self.schema.COLUMNS
        )

        return data_validation_config

    def get_data_split_config(self) -> DataSplitterConfig:
        config = self.config.data_splitter
        create_directories([config.root_dir])

        data_splitter_config = DataSplitterConfig(
            root_dir=config.root_dir,
            train_data=config.train_data,
            test_data=config.test_data,
            split_ratio=config.split_ratio,
            random_state=config.random_state,
            unzip_data_dir=config.unzip_data_dir,
            categorical_columns=self.schema.Categorial_COLUMNS,  # Match schema naming
            numerical_columns=self.schema.Numerical_COLUMNS      # Match schema naming
        )

        return data_splitter_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.RandomForest
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
            target_column=self.schema.TARGET_COLUMN
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_path=Path(config.metric_file_path),
            target_column=self.schema.TARGET_COLUMN,
            all_params=self.params.RandomForest,
            mlflow_uri="https://dagshub.com/suyashambule1234/End-to-End-IPL-prediction.mlflow"
        )

        return model_evaluation_config