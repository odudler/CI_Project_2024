import numpy as np
import pandas as pd
import math
import os
from sklearn.metrics import mean_squared_error


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

def create_submission(sub_sample_path='data/sampleSubmission.csv', store_path='submissions/default.csv', data_matrix):
    """
    Parameters:
        sub_sample_path (String): Path to sample submission file ("sampleSubmission.csv")
        store_path (String): Path to store submission file
        data_matrix (np.ndarray): Data matrix containing the predictions
    """
    df = pd.read_csv(sub_sample_path)
    users, movies, _ = extract_users_items_predictions(df)
    nrows = df.shape[0]

    df_sub = pd.DataFrame(columns=['row', 'col', 'pred'])
    df_sub['row'] = users + 1
    df_sub['col'] = movies + 1
    data_matrix = np.clip(data_matrix, 1, 5)
    df_sub['pred'] = data_matrix[users, movies]

    def construct_submission_format(df):
        return f"r{df['row']:.0f}_c{df['col']:.0f}"

    df_sub['entry'] = df_sub.apply(construct_submission_format, axis=1)
    df_sub = df_sub.drop(['row', 'col'], axis=1)
    df_sub.to_csv(store_path, columns=['entry', 'pred'], index=False)