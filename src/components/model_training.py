# Built-in
import sys

# Data Handling
import numpy as np
import pandas as pd

# Models
from sklearn.ensemble import RandomForestRegressor

# Custom
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException
from src.configs.variable_configs import ModelTrainerConfig


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_path, y_train_path):
        """
        This function handles the training of the model.
        :param X_train_path: X_train data path
        :param y_train_path: y_train data path
        :return: model
        """
        try:
            # Reading the data
            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path)

            logging.info("Training the initial model")
            model = RandomForestRegressor()
            model.fit(X_train, np.ravel(y_train))

            logging.info("Saving the initial model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return model

        except Exception as e:
            raise CustomException(e, sys)
