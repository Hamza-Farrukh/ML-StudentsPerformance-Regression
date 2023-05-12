# Built-in
import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_collection_path: str = 'data/raw/data.csv'
    analysis_folder_path: str = os.path.join('templates', 'analysis')
    eda_report_path: str = os.path.join('templates', 'analysis', 'eda_report.html')
    raw_data_path: str = os.path.join('data', 'raw', "data.csv")
    train_data_path: str = os.path.join('data', 'interim', "train.csv")
    test_data_path: str = os.path.join('data', 'interim', "test.csv")


@dataclass
class DataTransformationConfig:
    target_column_name: str = 'math_score'
    X_test_path: str = os.path.join('data', 'processed', 'X_test.csv')
    y_test_path: str = os.path.join('data', 'processed', 'y_test.csv')
    X_train_path: str = os.path.join('data', 'processed', 'X_train.csv')
    y_train_path: str = os.path.join('data', 'processed', 'y_train.csv')
    preprocessor_obj_file_path: str = os.path.join('models', 'preprocessor.pkl')
    features_configs_save_path: str = os.path.join('src', 'configs', 'features_configs.yml')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("models", "model.pkl")


@dataclass
class ModelEvaluationConfig:
    save_result_path = os.path.join('reports', 'result.txt')


@dataclass
class ModelParameterTuningConfig:
    tuned_model_file_path: str = os.path.join("models", "tuned_model.pkl")
    params = {
        'criterion': ['squared_error', 'absolute_error'],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }


@dataclass
class PredictConfig:
    trained_model_file_path: str = os.path.join("models", "model.pkl")
