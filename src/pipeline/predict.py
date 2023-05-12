# Custom
from src.utils import load_object
from src.configs.variable_configs import PredictConfig
from src.configs.variable_configs import ModelTrainerConfig
from src.configs.variable_configs import DataTransformationConfig


class Predict:
    def __init__(self):
        self.predict_configs = PredictConfig()

    def predict(self, X_input):
        """
        This function is responsible for prediction(s) from the data.
        :param X_input: Input values list
        :return: predictions
        """
        preprocessor = load_object(DataTransformationConfig().preprocessor_obj_file_path)
        model = load_object(ModelTrainerConfig().trained_model_file_path)
        # X_test = pd.DataFrame(
        #     [
        #         ['male', 'group C', 'associate\'s degree', 'standard', 'none', 86, 84]
        #     ],
        #     columns=[
        #         'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course',
        #         'reading score', 'writing score'
        #     ]
        # )
        X_input = preprocessor.transform(X_input)
        preds = model.predict(X_input)[0]

        return preds
