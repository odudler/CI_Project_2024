import yaml
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
from models import *
from utils import *

def load_yaml_parameters(filepath):
    '''
    Parameters:
        filepath (String): Path to file containing parameters
    Returns:
        data (dict): Contains all parameters 
    '''
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Generate all parameter combinations
def generate_param_combinations(param_grid):
    '''
    Parameters:
        param_grid (dict): contains all the parameters for the grid search
    Returns:
        Iterator over all possible combinations of parameters 
    '''
    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    for instance in product(*values):
        yield dict(zip(keys, instance))

#Get the proper model
def get_model(model_name, args):
    '''
    Parameters:
        model_name (String): Represents name of model we want to initialize
        args (argparse.Namespace): contains initalization arguments
    Returns:
        instance of a model of type "model_name" init. with args
    '''
    if model_name == "SVDplusplus":
        return SVDplusplus(args)
    elif model_name == "SVDsimple":
        return SVDsimple(args)
    elif model_name == "KNN":
        return KNN(args)
    elif model_name == "NeuralCF":
        return NeuralCF(args)
    else:
        raise ValueError(f"Model: {model_name} doesn't exist")

# Perform k-fold cross-validation for each parameter combination
def perform_grid_search(model_name, data, config_file, n_splits=5):
    '''
    Parameters:
        model_name(String): Represents name of model used
        data (df): Data to perform K-Fold cross-val. on
        config_file (String): path to grid-search parameters
    Returns:
        best_params (dict): Parameters leading to best score
        best_score (float): Best (lowest) achieved score
    '''
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
            #print(f"Score: {score}")
            scores.append(score)
            break #only do one fold for now, for computational reasons

        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

        print(f"Tested {params}, Score: {avg_score}")

    return best_params, best_score
