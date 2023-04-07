import os

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('data', 'raw', "data.csv")
    train_data_path: str = os.path.join('data', 'interim', "train.csv")
    test_data_path: str = os.path.join('data', 'interim', "test.csv")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('models', 'preprocessor.pkl')
    target_column_name = 'math score'
