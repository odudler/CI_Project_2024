import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from surprise import SVDpp, SVD, KNNWithZScore
from utils import (
    prepare_data_for_surprise,
    extract_users_items_predictions,
    create_submission_from_array,
    calculate_rmse,
)


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
