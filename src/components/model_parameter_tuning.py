# Built-in
import sys

# Data Handling
import numpy as np
import pandas as pd

# Selection
from sklearn.model_selection import GridSearchCV

# Custom
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException
from src.configs.variable_configs import ModelParameterTuningConfig


class ModelParameterTuning:
    def __init__(self):
        self.model_parameter_tuning_config = ModelParameterTuningConfig()

    def initiate_model_parameter_tuning(self, model, X_train_path, y_train_path):
        """
        This function is responsible for tuning the already created model.
        :param model: model object
        :param X_train_path: X_train data path
        :param y_train_path: y_train data path
        :return: tuned_model
        """
        try:
            logging.info("Initiating Hyperparameter Tuning")
            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path)

            # Hyperparameters
            tuner = GridSearchCV(
                estimator=model,
                param_grid=self.model_parameter_tuning_config.params,
                cv=3
            )
            tuner.fit(X_train, np.ravel(y_train))

            logging.info("Saving the tuned model")
            save_object(
                file_path=self.model_parameter_tuning_config.tuned_model_file_path,
                obj=tuner.best_estimator_
            )
            logging.info(f'Best parameters: {tuner.best_params_}')

            return tuner.best_estimator_

        except Exception as e:
            raise CustomException(e, sys)
