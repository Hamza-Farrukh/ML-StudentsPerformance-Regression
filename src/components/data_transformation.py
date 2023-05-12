# Built-in
import sys
import yaml

# Data Handling
import pandas as pd

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Transformers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Custom
from src.logger import logging
from src.exception import CustomException
from src.configs.variable_configs import DataTransformationConfig
from src.utils import save_object, separate_feature_types, organize_features


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformer_object(columns):
        """
        This function contains the data transformation pipeline.
        :return: preprocessor_object
        """
        try:
            continuous_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='mean')),
                    ('Scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                    ('Encoder', OneHotEncoder(sparse_output=False)),
                ]
            )

            logging.info(f"Continuous features: {columns['continuous']}")
            logging.info(f"Categorical features: {columns['categorical']}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('Continuous', continuous_pipeline, columns['continuous']),
                    ('Categorical', categorical_pipeline, columns['categorical'])
                ],
                remainder='drop'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        This function handles all the transformation on the train and test data.

        :param train_data_path: path of the training dataset
        :param test_data_path: path of the testing dataset
        :return: (X_train_path, y_train_path, X_test_path, y_test_path, preprocessor_obj_file_path)
        """
        try:
            logging.info("Reading the train and test data")
            train_data = pd.read_csv(train_data_path, na_filter=False)
            test_data = pd.read_csv(test_data_path, na_filter=False)
            target_column_name = self.data_transformation_config.target_column_name

            # Splitting the training and target features
            X_train = train_data.drop(columns=[target_column_name], axis=1)
            y_train = train_data[target_column_name]

            X_test = test_data.drop(columns=[target_column_name], axis=1)
            y_test = test_data[target_column_name]

            # Getting the list of features and their types and organizing the features
            columns = separate_feature_types(X_train)
            features_list = organize_features(X_train, columns)

            logging.info("Obtaining the preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(columns)

            logging.info(f"Applying preprocessing object on training and testing dataframe")
            X_train = pd.DataFrame(preprocessing_obj.fit_transform(X_train))
            X_test = pd.DataFrame(preprocessing_obj.transform(X_test))

            # Saving the features in yaml config file
            features_dict = {
                'features_in': preprocessing_obj.feature_names_in_.tolist(),
                'columns': features_list
            }
            with open(self.data_transformation_config.features_configs_save_path, 'w') as f:
                yaml.dump(features_dict, f)

            # Saving the transformed train and test data
            X_train.to_csv(self.data_transformation_config.X_train_path, index=False)
            y_train.to_csv(self.data_transformation_config.y_train_path, index=False)
            X_test.to_csv(self.data_transformation_config.X_test_path, index=False)
            y_test.to_csv(self.data_transformation_config.y_test_path, index=False)

            logging.info(f"Saving the preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                self.data_transformation_config.X_train_path,
                self.data_transformation_config.y_train_path,
                self.data_transformation_config.X_test_path,
                self.data_transformation_config.y_test_path,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
