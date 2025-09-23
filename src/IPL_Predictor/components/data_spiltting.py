from src.IPL_Predictor.constants import *
from src.IPL_Predictor.utils.common import read_yaml, create_directories
from src.IPL_Predictor import logger
from src.IPL_Predictor.entity.config_entity import DataIngestionConfig, DataSplitterConfig
import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split    



class Data_Splitting:
    def __init__(self, config: DataSplitterConfig):
        self.config = config

    def split_data(self):
        df=pd.read_csv(self.config.unzip_data_dir)
        train_set, test_set = train_test_split(df, test_size=self.config.split_ratio, random_state=self.config.random_state)
        train_set.to_csv(self.config.train_data, index=False)
        test_set.to_csv(self.config.test_data, index=False)
        logger.info(f"Train and Test data saved in {self.config.root_dir}")
        logger.info(f"Train data shape: {train_set.shape} and Test data shape: {test_set.shape}")

