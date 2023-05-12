# Custom
from src.components.model_training import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.model_evaluation import ModelEvaluation
from src.components.data_transformation import DataTransformation
from src.components.model_parameter_tuning import ModelParameterTuning


if __name__ == "__main__":
    data_injestion = DataIngestion()
    train_data_path, test_data_path = data_injestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train_path, y_train_path, X_test_path, y_test_path, _ = data_transformation.initiate_data_transformation(
        train_data_path,
        test_data_path,
    )
