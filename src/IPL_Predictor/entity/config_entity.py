from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE:str
    unzip_data_dir:Path
    all_schema:dict
    
@dataclass
class DataSplitterConfig:  
    root_dir: Path
    train_data: Path
    test_data: Path
    split_ratio: float
    random_state: int
    unzip_data_dir: Path 
    categorical_columns: list 
    numerical_columns: list  

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    n_estimators: int
    max_depth: int
    random_state: int
    n_jobs: int
    target_column: str 

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_path: Path
    target_column: str
    all_params: dict
    mlflow_uri: str
