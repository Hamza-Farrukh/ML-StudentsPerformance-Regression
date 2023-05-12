# Built-in
import yaml

# Data Handling
import pandas as pd

# Web
from django.shortcuts import render

# Custom
from src.pipeline.predict import Predict


def index(request):
    with open('src/configs/features_configs.yml', 'r') as f:
        configs = yaml.safe_load(f)
        features = configs['columns']
    params = {}
    params.update(features)
    return render(request, 'home/index.html', params)


def results(request):
    with open('src/configs/features_configs.yml', 'r') as f:
        configs = yaml.safe_load(f)
        features = configs['columns']
        features_in = configs['features_in']
    params = {}
    params.update(features)
    values_list = {}

    for feature in features_in:
        values_list[feature] = request.GET.get(feature)

    input_values = pd.DataFrame(values_list, index=[0])
    result = Predict().predict(input_values)

    params['result'] = result
    params['features'] = zip(input_values.columns, input_values.values[0])

    return render(request, 'students_performance/results.html', params)


def eda(request):
    return render(request, 'analysis/eda.html')


def about(request):
    return render(request, 'about/about.html')
