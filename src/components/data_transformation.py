import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.configs import DataTransformationConfig


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformer_object(raw_data_path):
        """
        This function is responsible for the data transformation
        :return:
        """
        try:
            data = pd.read_csv(raw_data_path)
            numerical_columns = [column for column in data if data[column].dtype != 'O']
            categorical_columns = [column for column in data if data[column].dtype == 'O']

            # num_pipeline = Pipeline(
            #     steps=[
            #         ('Imputer', SimpleImputer(strategy='mean')),
            #         ('Scaler', StandardScaler())
            #     ]
            # )

            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                    ('Encoder', OneHotEncoder(sparse_output=False)),
                ]
            )

            logging.info(f"Numerical features: {numerical_columns}")
            logging.info(f"Categorical features: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    # ('Numerical', num_pipeline, numerical_columns),
                    ('Categorical', cat_pipeline, categorical_columns)
                ],
                remainder='drop'
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_path, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Reading the train and test data")
            logging.info("Obtaining the preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(raw_data_path=raw_path)

            target_column_name = DataTransformationConfig.target_column_name

            x_train = train_data.drop(columns=[target_column_name], axis=1)
            y_train = train_data[target_column_name]

            x_test = test_data.drop(columns=[target_column_name], axis=1)
            y_test = test_data[target_column_name]

            logging.info(
                f"Applying preprocessing object on training and testing dataframe"
            )

            x_train_arr = preprocessing_obj.fit_transform(x_train)
            x_test_arr = preprocessing_obj.transform(x_test)

            train_arr = np.c_[
                x_train_arr, np.array(y_train)
            ]
            test_arr = np.c_[
                x_test_arr, np.array(y_test)
            ]

            logging.info(f"Saved the preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                x_train_arr,
                # np.array(y_train),
                x_test_arr,
                # np.array(y_test),
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
