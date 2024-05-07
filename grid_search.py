import yaml
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
from models import *
from utils import *

def load_yaml_parameters(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Generate all parameter combinations
def generate_param_combinations(param_grid):
    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    for instance in product(*values):
        yield dict(zip(keys, instance))

#Get the proper model
def get_model(model_name, args):
    if model_name == "SVDplusplus":
        return SVDplusplus(args)
    elif model_name == "":
        return None


# Perform k-fold cross-validation for each parameter combination
def perform_grid_search(model_name, data, config_file, n_splits=5):

    config = load_yaml_parameters(config_file)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_score = float('inf')
    best_params = None
    

    for params in generate_param_combinations(config[model_name]):
        print(f"Testing {params}")
        scores = []
        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            args = set_args(params, model_name)
            model = get_model(model_name, args)
            model.train(train_data)
            score = model.predict(test_data, return_loss=True)
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

        print(f"Tested {params}, Score: {avg_score}")

    return best_params, best_score
