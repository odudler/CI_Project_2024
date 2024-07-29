import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from surprise import Reader, Dataset
import argparse
import yaml

def extract_users_items_predictions(df):
    """
    Parameters:
        df (pd.DataFrame): DataFrame to process, containing users, movie and prediction information
    Returns:
        users, movies, predictions (np.array): Array of user and movie ids and the corresponding predictions.

    User/Movie indices started from 1, now corrected to 0
    """
    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            df.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1
        )
    ]
    predictions = df.Prediction.values
    return users, movies, predictions
    

def normalize_columns(data, n_users, n_movies, mask_value=0):
    """
    Parameters:
        data (np.ndarray): two-dimensional data matrix containing ratings
        n_users (int): numner of users/rows in data matrix
        n_movies (int): number of movies/columns in data matrix
        mask_value (int): value to mask in data matrix
    Returns:
        data_normalized (np.ndarray): two-dimensional data matrix containing normalized ratings
        mean (np.ndarray): two-dimensional data matrix containing data mean
        std: data standard deviation
    """
    mask = np.ma.masked_equal(data, mask_value)
    # to check: mean along row / col have effects on results?
    mean = np.tile(np.ma.mean(mask, axis=0).data, (n_users, 1))
    std = np.tile(np.ma.std(mask, axis=0).data, (n_users, 1))
    data_normalized = ((mask - mean) / std).data
    return data_normalized, mean, std

def denormalize_columns(data, mean, std):
    """
    given the data mean and std, undoes data normalization
    """
    return np.clip(data * std + mean, 1, 5)
    

def prepare_data_for_surprise(df):
    """
    Parameters:
        df (pd.DataFrame): DataFrame to process, containing users, movie and prediction information
    Returns:
        dataset (surprise.Dataset): Dataset containing the training data
    """
    users, movies, predictions = extract_users_items_predictions(df)
    df_surprise = pd.DataFrame(
        {"itemID": movies, "userID": users, "rating": predictions}
    )
    return Dataset.load_from_df(df_surprise, reader=Reader(rating_scale=(1, 5)))

def prepare_data_for_surprise_nondf(users, movies, predictions):
    """
    Parameters:
        users, movies, predictions (np.array): Array of user and movie ids and the corresponding predictions.
    Returns:
        dataset (surprise.Dataset): Dataset containing the training data
    """
    df_surprise = pd.DataFrame(
        {"itemID": movies, "userID": users, "rating": predictions}
    )
    return Dataset.load_from_df(df_surprise, reader=Reader(rating_scale=(-3, 3)))

def prepare_data_for_BFM(df):
    users, movies, predictions = extract_users_items_predictions(df)
    df_new = pd.DataFrame({
        'user_id': users,
        'movie_id': movies,
        'rating': predictions
    })
    return df_new

def create_data_matrix(users, movies, predictions, num_users=10000, num_movies=1000):
    """
    Parameters:
        users, movies, predictions (np.array): Array of user and movie ids and the corresponding predictions.
        num_users, num_movies (int. optional): number of rows/columns of the matrix

    Returns:
        matrix (np.ndarray): Data Matrix containing predictions
    """
    matrix = np.zeros((num_users, num_movies), dtype=np.float16)
    matrix[users, movies] = predictions.astype(np.float16)
    return matrix

def convert_matrix_to_data(data_matrix, masked_value):
#      Find indices where the value is not "masked_value"
    users, movies = np.where(data_matrix != masked_value)
    
    # Get the corresponding ratings
    predictions = data_matrix[users, movies]
    
    return users, movies, predictions

def convert_matrix_to_data_given_data(data_matrix, users, movies):
    """
    extracts data from data matrix to give sparse arrays
    """
    predictions = data_matrix[users, movies]

    return users, movies, predictions



def create_submission_from_matrix(
    data_matrix, users, movies, store_path="submissions/default.csv"
):
    """
    Parameters:
        users (np.array): List of user IDs to predict rating of
        movies (np.array): List of movie IDs to predict rating of
        store_path (String): Path to store submission file
        data_matrix (np.ndarray): Data matrix containing rating predictions
    """

    df_sub = pd.DataFrame(columns=["row", "col", "pred"])
    df_sub["row"] = users + 1
    df_sub["col"] = movies + 1
    data_matrix = np.clip(data_matrix, 1, 5)
    df_sub["Prediction"] = data_matrix[users, movies]

    def construct_submission_format(df):
        return f"r{df['row']:.0f}_c{df['col']:.0f}"

    df_sub["Id"] = df_sub.apply(construct_submission_format, axis=1)
    df_sub = df_sub.drop(["row", "col"], axis=1)
    df_sub.to_csv(store_path, columns=["Id", "Prediction"], index=False)


def create_submission_from_array(
    predictions, users, movies, store_path="submissions/default.csv"
):
    """
    Parameters:
        users (np.array): List of user IDs to predict rating of
        movies (np.array): List of movie IDs to predict rating of
        store_path (String): Path to store submission file
        predictions (np.array): Array containing rating predictions
    """

    df_sub = pd.DataFrame(columns=["row", "col", "pred"])
    df_sub["row"] = users + 1
    df_sub["col"] = movies + 1
    df_sub["Prediction"] = predictions

    def construct_submission_format(df):
        return f"r{df['row']:.0f}_c{df['col']:.0f}"

    df_sub["Id"] = df_sub.apply(construct_submission_format, axis=1)
    df_sub = df_sub.drop(["row", "col"], axis=1)
    df_sub.to_csv(store_path, columns=["Id", "Prediction"], index=False)

def create_ensemble(paths, store_path="submissions/default.csv"):
    """
    Parameters:
        paths (List): List of paths to submission files to ensemble
        store_path (String): Path to store submission file
    """
    df = pd.read_csv(paths[0])
    df["Prediction"] = 0
    for path in paths:
        df["Prediction"] += pd.read_csv(path)["Prediction"]
    df["Prediction"] /= len(paths)
    df.to_csv(store_path, columns=["Id", "Prediction"], index=False)

def calculate_rmse(preds, labels):
    """
    Parameters:
        preds (np.array): predictions of ratings
        labels (np.array): true ratings by users

    Returns:
        rmse_loss (float): Rooted mean squared error
    """
    return math.sqrt(mean_squared_error(preds, labels))


def read_config(config_path):
    """
    Parameters:
        config_path (String): path to the config to load

    Returns:
        args: argument Namespace containing the config arguments
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file.read())

    return argparse.Namespace(
        svdpp=argparse.Namespace(**(config["args"]["SVDplusplus"] or {})),
        svd=argparse.Namespace(**(config["args"]["SVDsimple"] or {})),
        knn=argparse.Namespace(**(config["args"]["KNN"] or {})),
        bfm=argparse.Namespace(**(config["args"]["BFM"] or {})),
        **(config["args"]["training"] or {}),
    )


def set_args(params, model_name, config_path="config_models.yaml"):
    """
    This function sets the arguments to be passed into the initialization function of a model

    Parameters:
        params (dict): arguments for model initialization
        model_name (String): Name of the model to be initialized
        config_path (String): path to the config to load

    Returns:
        args: arguments to be passed into model initialization
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file.read())

    args = argparse.Namespace()

    # General training arguments
    args.random_state = config["args"]["training"]["random_state"]
    args.n_users = config["args"]["training"]["n_users"]
    args.n_movies = config["args"]["training"]["n_movies"]
    args.test_set_size = config["args"]["training"]["test_set_size"]
    args.verbose = config["args"]["training"]["verbose"]

    if model_name == "SVDplusplus":
        args.svdpp = argparse.Namespace()
        args.svdpp.n_factors = params["n_factors"]
        args.svdpp.lr_all = params["lr_all"]
        args.svdpp.n_epochs = params["n_epochs"]
        args.svdpp.reg_all = params["reg_all"]

    if model_name == "SVDsimple":
        args.svd = argparse.Namespace()
        args.svd.n_factors = params["n_factors"]
        args.svd.lr_all = params["lr_all"]
        args.svd.n_epochs = params["n_epochs"]
        args.svd.reg_all = params["reg_all"]

    if model_name == "KNN":
        args.knn = argparse.Namespace()
        args.knn.k = params["k"]
        args.knn.min_k = params["min_k"]
        args.knn.sim_options = params["sim_options"]

    if model_name == "BFM":
        args.bfm = argparse.Namespace()
        args.bfm.algorithm = params['algorithm']
        args.bfm.variational = params['variational']
        args.bfm.iteration = params['iteration']
        args.bfm.dimension = params['dimension']
        args.bfm.use_iu = params['use_iu']
        args.bfm.use_ii = params['use_ii']


    return args


if __name__ == "__main__":
    args = read_config("config_models.yaml")
