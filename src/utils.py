# Built-in
import os
import sys
import pickle

# Data Handling
import pandas as pd

# File Handling
import dill

# Custom
from src.exception import CustomException


def save_object(file_path, obj):
    """
    This function saves the object files to the specified path.

    :param file_path: obj_file_path
    :param obj: object to be saved
    :returns: None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    This function loads the object file from the specified path.

    :param file_path: obj_file_path
    :returns: None
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def clean_feature_names(data):
    """
    This function cleans the feature names of the given pandas dataframe.
    Following things are done:

    - space replaced by "_"
    - values of categorical features are changed as title string

    :param data: pandas dataframe to be used
    :return: cleaned_data
    """
    # Cleaning feature names
    clean_names = []
    for name in data.columns:
        name = name.lower()
        name = name.replace(' ', '_')
        clean_names.append(name)
    data.columns = clean_names

    # Cleaning categorical feature names
    categorical = [column for column in data if data[column].dtype == 'O']
    for column in data[categorical].columns:
        list_unique = data[column].unique().tolist()
        new_names = [unique.title() for unique in list_unique]
        mapping = {old: new for old, new in zip(list_unique, new_names)}
        data[column] = data[column].map(mapping, na_action='ignore')
    return data


def separate_feature_types(data):
    """
    This function separates the feature names according to their type as follows:

    - continuous
    - discrete
    - categorical
    - time (make sure that the data type is datetime[ns])

    :param data: dataframe
    :return: {'type_name': features_list} for all types in same dict
    """
    # Storing categorical features
    categorical = [column for column in data if data[column].dtype == 'O']

    # Storing all continuous features
    continuous = [
        column for column in data
        if ((data[column].dtype == 'f8') or (data[column].dtype == 'i8')) and (data[column].nunique() > 10)
    ]

    # Storing all discrete features
    discrete = [
        column for column in data
        if ((data[column].dtype == 'f8') or (data[column].dtype == 'i8')) and (data[column].nunique() < 10)
    ]

    # Storing all time features
    time = [column for column in data if data[column].dtype == 'datetime64']

    # Returning the feature names
    return {'continuous': continuous, 'discrete': discrete, 'categorical': categorical, 'time': time}


def organize_features(X_train, columns):
    """
    This function organizes the features according to following format:

    {feature_type: [feature_name, feature_name without the "_", unique_values(categorical and discrete only)]

    :param X_train: training data
    :param columns: > dictionary containing the column names separated by types
    :return: dictionary containing the organized features
    """
    categorical_unique = {}
    for feature in columns['categorical']:
        categorical_unique[feature] = sorted(X_train[feature].unique().tolist())

    discrete_unique = {}
    for feature in columns['discrete']:
        discrete_unique[feature] = X_train[feature].unique().tolist()

    features = {}
    for type in columns:
        if type == 'categorical':
            features[type] = [
                [feature, feature.replace('_', ' ').title(), categorical_unique[feature]]
                for feature in columns[type]
            ]
        elif type == 'discrete':
            features[type] = [
                [feature, feature.replace('_', ' ').title(), discrete_unique[feature]]
                for feature in columns[type]
            ]
        else:
            features[type] = [
                [feature, feature.replace('_', ' ').title()] for feature in columns[type]
            ]

    return features
