# Built-in
import os
import sys
import pickle

# File handling
import dill

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import r2_score

# Custom
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            # Searching best parameters
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            # Setting best parameter and Training the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # y_train_pred = model.predict(X_train)
            # train_model_score = r2_score(y_train, y_train_pred)

            # Evaluating with test data
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
