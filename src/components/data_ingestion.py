# Built-in
import os
import sys

# Data Handling
import pandas as pd
from pandas_profiling import ProfileReport

# Selection
from sklearn.model_selection import train_test_split

# Custom
from src.logger import logging
from src.utils import clean_feature_names
from src.exception import CustomException
from src.configs.variable_configs import DataIngestionConfig


class DataIngestion:
    """
    This class is responsible for loading the from the sources i.e. database, files etc.
    This class also generates a report of EDA using the pandas-profiling library.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function collects data then split it into train and test set and saves them.
        It also generates an EDA report.

        :return: (train_data_path, test_data_path)
        """

        logging.info("Initiating the data injestion")
        try:
            logging.info('Reading the dataset')
            data = pd.read_csv(self.ingestion_config.data_collection_path)

            # Making directories if they don't exist already
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.analysis_folder_path), exist_ok=True)

            logging.info('Making the EDA report')
            profile = ProfileReport(
                df=data,
                explorative=True,
                config_file=os.path.join('src/configs/html_configs.yml'),
            )
            profile.config.html.full_width = True
            profile.config.html.navbar_show = False
            profile.config.html.style.theme = None
            profile.to_file(self.ingestion_config.eda_report_path)

            # Saving the raw data
            data.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            # Cleaning the feature names
            data = clean_feature_names(data)

            logging.info("Splitting the data into train and test")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
