# Built-in
import sys

# Data handling
import pandas as pd

# Metrics
from sklearn.metrics import r2_score

# Custom
from src.logger import logging
from src.exception import CustomException
from src.configs.variable_configs import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, model, X_test_path, y_test_path):
        """
        This function handles the evaluation of the model.
        :param model: model object
        :param X_test_path: X_test data path
        :param y_test_path: y_test data path
        :return: accuracy_result
        """
        try:
            logging.info("Evaluating the model")

            # Getting the data
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path)

            # Predicting
            y_pred = model.predict(X_test)

            # Calculating the result
            result = r2_score(y_test, y_pred)

            logging.info(f"Result: {result}")

            return result

        except Exception as e:
            raise CustomException(e, sys)
