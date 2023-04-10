# Custom
from src.components.model_training import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


if __name__ == "__main__":
    data_injestion = DataIngestion()
    raw_data, train_data, test_data = data_injestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        raw_data,
        train_data,
        test_data,
    )
    model_training = ModelTrainer()
    result = model_training.initiate_model_trainer(train_arr, test_arr)
