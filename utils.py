import numpy as np
import pandas as pd
import math
import os


def extract_users_items_predictions(df):
    """
    Parameters:
        df (pd.DataFrame): DataFrame to process, containing users, movie and prediction information
    Returns:
        users, movies, predictions (np.array): Array of user and movie ids and the corresponding predictions.

    User/Movie indices started from 1, now corrected to 0
    """
    users, movies = \
        [np.squeeze(arr) for arr in np.split(df.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = df.Prediction.values
    return users, movies, predictions

def create_data_matrix(users, movies, predictions, num_users=10000, num_movies=1000):
    """
    Parameters:
        users, movies, predictions (np.array): Array of user and movie ids and the corresponding predictions.
        num_users, num_movies (int. optional): number of rows/columns of the matrix
    
    Returns:
        matrix (np.ndarray): Data Matrix containing predictions
    """
    #TODO: is float16 enough?
    matrix = np.zeros((num_users, num_movies), dtype=np.float16)
    matrix[users, movies] = predictions.astype(np.float16)
    return matrix

