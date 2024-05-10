import numpy as np
import os
import itertools as IT
from tqdm import tqdm
from surprise import SVDpp, SVD, KNNWithZScore
from utils import (
    prepare_data_for_surprise,
    prepare_data_for_recommender,
    extract_users_items_predictions,
    create_submission_from_array,
    create_submission_from_matrix,
    calculate_rmse,
    denormalize_columns
)

from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset


class SVDplusplus:
    def __init__(self, args):
        self.svdpp_args = args.svdpp
        self.verbose = args.verbose
        self.random_state = args.random_state

    def train(self, df):
        dataset = prepare_data_for_surprise(df)
        trainset = dataset.build_full_trainset()
        svdpp = SVDpp(
            **self.svdpp_args.__dict__,
            verbose=self.verbose,
            random_state=self.random_state,
            cache_ratings=True,  # Should speed up computation
        )
        svdpp.fit(trainset)
        self.model = svdpp

    def predict(self, df, output_file=None, return_loss=False):
        users, movies, labels = extract_users_items_predictions(df)
        predictions = np.empty(len(labels))

        for i in tqdm(range(len(users)), desc="Prediction Loop"):
            predictions[i] = self.model.predict(
                users[i], movies[i], verbose=self.verbose
            ).est

        if output_file is not None:
            create_submission_from_array(predictions, users, movies, output_file)

        loss = calculate_rmse(predictions, labels)
        if return_loss:
            return loss
        else:
            print(f"RMSE for {self.__class__.__name__}: {loss}")
            return None


class SVDsimple:
    def __init__(self, args):
        self.svd_args = args.svd
        self.verbose = args.verbose
        self.random_state = args.random_state

    def train(self, df):
        dataset = prepare_data_for_surprise(df)
        trainset = dataset.build_full_trainset()
        svd = SVD(
            **self.svd_args.__dict__,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        svd.fit(trainset)
        self.model = svd

    def predict(self, df, output_file=None, return_loss=False):
        users, movies, labels = extract_users_items_predictions(df)
        predictions = np.empty(len(labels))

        for i in tqdm(range(len(users)), desc="Prediction Loop"):
            predictions[i] = self.model.predict(
                users[i], movies[i], verbose=self.verbose
            ).est

        if output_file is not None:
            create_submission_from_array(predictions, users, movies, output_file)

        loss = calculate_rmse(predictions, labels)
        if return_loss:
            return loss
        else:
            print(f"RMSE for {self.__class__.__name__}: {loss}")
            return None


class KNN:
    def __init__(self, args):
        self.knn_args = args.knn
        self.verbose = args.verbose

    def train(self, df):
        dataset = prepare_data_for_surprise(df)
        trainset = dataset.build_full_trainset()
        knn = KNNWithZScore(**self.knn_args.__dict__, verbose=self.verbose)
        knn.fit(trainset)
        self.model = knn

    def predict(self, df, output_file=None, return_loss=False):
        users, movies, labels = extract_users_items_predictions(df)
        predictions = np.empty(len(labels))

        for i in tqdm(range(len(users)), desc="Prediction Loop"):
            predictions[i] = self.model.predict(
                users[i], movies[i], verbose=self.verbose
            ).est

        if output_file is not None:
            create_submission_from_array(predictions, users, movies, output_file)

        loss = calculate_rmse(predictions, labels)
        if return_loss:
            return loss
        else:
            print(f"RMSE for {self.__class__.__name__}: {loss}")
            return None



class NeuralCF:
    def __init__(self, args):
        self.ncf_args = args.ncf
        self.verbose = args.verbose
        self.random_state=args.random_state
        self.n_users = args.n_users
        self.n_movies = args.n_movies

    def train(self, df):
        #Need to create temporary file with data as NCFDataset expects file
        self.mean, self.std = prepare_data_for_recommender(df, self.n_users, self.n_movies, 'temp.csv')
        data = NCFDataset(train_file='temp.csv', test_file=None, seed=self.random_state, binary=False,
                          overwrite_test_file_full=False, n_neg_test=0, n_neg=0)
        
        model = NCF (
            n_users=self.n_users, 
            n_items=self.n_movies,
            **self.ncf_args.__dict__,
            verbose=self.verbose,
            seed=self.random_state
        )

        model.fit(data)
        self.model = model

        #Delete temporary file again
        if os.path.isfile('temp.csv'):
            os.remove('temp.csv')

    def predict(self, df, output_file=None, return_loss=False):
        users, movies, labels = extract_users_items_predictions(df)
        predictions = np.empty(len(labels))

        reconstructed = np.zeros((self.n_users, self.n_movies))
        idx = 0
        for i, j in IT.izip(users, movies):
            reconstructed[i,j] = predictions[idx] = self.model.predict(i,j)
            idx+=1
        reconstructed = denormalize_columns(reconstructed, self.mean, self.std)

        if output_file is not None:
            create_submission_from_matrix(reconstructed, users, movies, output_file)

        loss = calculate_rmse(predictions, labels)
        if return_loss:
            return loss
        else:
            print(f"RMSE for {self.__class__.__name__}: {loss}")
            return None