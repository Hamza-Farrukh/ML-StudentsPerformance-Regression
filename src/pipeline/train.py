# Custom
from src.components.model_training import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.model_evaluation import ModelEvaluation
from src.components.data_transformation import DataTransformation
from src.components.model_parameter_tuning import ModelParameterTuning


if __name__ == "__main__":
    # Class objects
    data_injestion = DataIngestion()
    data_transformation = DataTransformation()
    model_training = ModelTrainer()
    model_evaluation = ModelEvaluation()

    # Data Injestion
    train_data_path, test_data_path = data_injestion.initiate_data_ingestion()

    # Data Transformation
    X_train_path, y_train_path, X_test_path, y_test_path, _ = data_transformation.initiate_data_transformation(
        train_data_path,
        test_data_path,
    )

    # Model Training
    model = model_training.initiate_model_trainer(X_train_path, y_train_path)

    # Model Evaluation
    model_result = model_evaluation.initiate_model_evaluation(model, X_test_path, y_test_path)

    # Model Tuning
    model_parameter_tuning = ModelParameterTuning()
    tuned_model = model_parameter_tuning.initiate_model_parameter_tuning(model, X_train_path, y_train_path)

    # Tuned Model Evaluation
    tuned_model_result = model_evaluation.initiate_model_evaluation(tuned_model, X_test_path, y_test_path)
