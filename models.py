import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Union
from scipy import sparse as sps

from utils import (
    prepare_data_for_surprise,
    prepare_data_for_recommender,
    prepare_data_for_BFM,
    extract_users_items_predictions,
    create_submission_from_array,
    create_submission_from_matrix,
    calculate_rmse,
    denormalize_columns,
    prepare_data_for_surprise_nondf,
    create_data_matrix,
    normalize_columns,
    convert_matrix_to_data,
    convert_matrix_to_data_given_data
)
#SVD + KNN imports
from surprise import SVDpp, SVD, KNNWithZScore
#NCF imports
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
#BMF imports
import myfm
from myfm import RelationBlock, MyFMOrderedProbit, MyFMRegressor
from myfm.utils.encoders import CategoryValueToSparseEncoder
from myfm.gibbs import MyFMOrderedProbit
from myfm.utils.callbacks.libfm import LibFMLikeCallbackBase

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
            cache_ratings=True,
        )
        svdpp.fit(trainset)
        self.model = svdpp

    # def train(self, df):
    #     users, movies, predictions = extract_users_items_predictions(df)
    #     data_matrix = create_data_matrix(users, movies, predictions)
    #     data_matrix[data_matrix == 0] = -10
    #     data_norm, self.mean, self.std = normalize_columns(data_matrix, 10000, 1000, mask_value=-10)
    #     users, movies, predictions = convert_matrix_to_data(data_norm, -10)

    #     dataset = prepare_data_for_surprise_nondf(users, movies, predictions)
    #     trainset = dataset.build_full_trainset()
    #     svdpp = SVDpp(
    #         **self.svdpp_args.__dict__,
    #         verbose=self.verbose,
    #         random_state=self.random_state,
    #         cache_ratings=True,
    #     )
    #     svdpp.fit(trainset)
    #     self.model = svdpp

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

    # def predict(self, df, output_file=None, return_loss=False):
    #     users, movies, labels = extract_users_items_predictions(df)
    #     predictions = np.empty(len(labels))

    #     for i in tqdm(range(len(users)), desc="Prediction Loop"):
    #         predictions[i] = self.model.predict(
    #             users[i], movies[i], verbose=self.verbose
    #         ).est

    #     if output_file is not None:
    #         data_matrix = create_data_matrix(users, movies, predictions)
    #         data_denorm = denormalize_columns(data_matrix, self.mean, self.std)
    #         users, movies, predictions = convert_matrix_to_data_given_data(data_denorm, users, movies)


    #         create_submission_from_array(predictions, users, movies, output_file)

    #     loss = calculate_rmse(predictions, labels)
    #     if return_loss:
    #         return loss
    #     else:
    #         print(f"RMSE for {self.__class__.__name__}: {loss}")
    #         return None


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
    #Code inspired by: https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb
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
        for i, j in zip(users, movies):
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
        
class BFM:
    #Code inspired by: https://github.com/tohtsky/myFM/blob/main/examples/ml-100k-regression.py
    def __init__(self, args):
        self.bfm_args = args.bfm
        self.random_state = args.random_state
        self.fm = None
        self.movie_vs_watched = None
        self.user_vs_watched = None
        self.movie_to_internal = None
        self.user_to_internal = None

    # given user/movie ids, add additional infos and return it as sparse
    def augment_user_id(self, user_ids: List[int]) -> sps.csr_matrix:
            X = self.user_to_internal.to_sparse(user_ids)
            if not self.bfm_args.use_iu:
                return X
            data: List[float] = []
            row: List[int] = []
            col: List[int] = []
            for index, user_id in enumerate(user_ids):
                watched_movies = self.user_vs_watched.get(user_id, [])
                normalizer = 1 / max(len(watched_movies), 1) ** 0.5
                for mid in watched_movies:
                    data.append(normalizer)
                    col.append(self.movie_to_internal[mid])
                    row.append(index)
            return sps.hstack(
                [
                    X,
                    sps.csr_matrix(
                        (data, (row, col)),
                        shape=(len(user_ids), len(self.movie_to_internal)),
                    ),
                ],
                format="csr",
            )
    
    def augment_movie_id(self, movie_ids: List[int]):
            X = self.movie_to_internal.to_sparse(movie_ids)
            if not self.bfm_args.use_ii:
                return X

            data: List[float] = []
            row: List[int] = []
            col: List[int] = []

            for index, movie_id in enumerate(movie_ids):
                watched_users = self.movie_vs_watched.get(movie_id, [])
                normalizer = 1 / max(len(watched_users), 1) ** 0.5
                for uid in watched_users:
                    data.append(normalizer)
                    row.append(index)
                    col.append(self.user_to_internal[uid])
            return sps.hstack(
                [
                    X,
                    sps.csr_matrix(
                        (data, (row, col)),
                        shape=(len(movie_ids), len(self.user_to_internal)),
                    ),
                ]
            )
    
    def train(self, df_train):
        np.random.seed(self.random_state)
        df_train = prepare_data_for_BFM(df_train)
        
        if self.bfm_args.algorithm == "oprobit":
            # interpret the rating (1, 2, 3, 4, 5) as class (0, 1, 2, 3, 4).
            for df_ in [df_train]:
                df_["rating"] -= 1
                df_["rating"] = df_.rating.astype(np.int32)

        implicit_data_source = df_train
        user_to_internal = CategoryValueToSparseEncoder[int](
            implicit_data_source.user_id.values
        )
        movie_to_internal = CategoryValueToSparseEncoder[int](
            implicit_data_source.movie_id.values
        )

        self.movie_to_internal = movie_to_internal
        self.user_to_internal = user_to_internal

        print(
            "df_train.shape = {}".format(df_train.shape)
        )

        movie_vs_watched: Dict[int, List[int]] = dict()
        user_vs_watched: Dict[int, List[int]] = dict()

        for row in implicit_data_source.itertuples():
            user_id = row.user_id
            movie_id = row.movie_id
            movie_vs_watched.setdefault(movie_id, list()).append(user_id)
            user_vs_watched.setdefault(user_id, list()).append(movie_id)

        self.movie_vs_watched = movie_vs_watched
        self.user_vs_watched = user_vs_watched

        # setup grouping
        feature_group_sizes = []

        feature_group_sizes.append(len(user_to_internal))  # user ids

        if self.bfm_args.use_iu:
            # all movies which a user watched
            feature_group_sizes.append(len(movie_to_internal))

        feature_group_sizes.append(len(movie_to_internal))  # movie ids

        if self.bfm_args.use_ii:
            feature_group_sizes.append(
                len(user_to_internal)  # all the users who watched a movies
            )

        grouping = [i for i, size in enumerate(feature_group_sizes) for _ in range(size)]

        train_blocks: List[RelationBlock] = []
        for source, target in [(df_train, train_blocks)]:
            unique_users, user_map = np.unique(source.user_id, return_inverse=True)
            target.append(RelationBlock(user_map, self.augment_user_id(unique_users)))
            unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
            target.append(RelationBlock(movie_map, self.augment_movie_id(unique_movies)))

        if self.bfm_args.algorithm == "regression":
            if self.bfm_args.variational:
                fm = myfm.VariationalFMRegressor(rank=self.bfm_args.dimension, random_seed=self.random_state)
            else:
                fm = myfm.MyFMRegressor(rank=self.bfm_args.dimension, random_seed=self.random_state)
        elif self.bfm_args.algorithm == "oprobit":
            fm = myfm.MyFMOrderedProbit(rank=self.bfm_args.dimension, random_seed=self.random_state)

        fm.fit(
            None,
            df_train.rating.values,
            X_rel=train_blocks,
            grouping=grouping,
            n_iter=self.bfm_args.iteration
        )
        self.fm = fm

    def predict(self, df, output_file=None, return_loss= False):
        df = prepare_data_for_BFM(df)
        users = df['user_id']
        movies = df['movie_id']
        labels = df['rating']

        blocks: List[RelationBlock] = []

        unique_users, user_map = np.unique(df.user_id, return_inverse=True)
        blocks.append(RelationBlock(user_map, self.augment_user_id(unique_users)))
        unique_movies, movie_map = np.unique(df.movie_id, return_inverse=True)
        blocks.append(RelationBlock(movie_map, self.augment_movie_id(unique_movies)))

        result = None
        if self.bfm_args.algorithm == "regression":
            result = (self.fm.predict(None, X_rel=blocks)).clip(1, 5)
        else:
            result = (self.fm.predict_proba(None, X_rel=blocks).dot(np.arange(5)) + 1).clip(1, 5)

        if output_file is not None:
            create_submission_from_array(result, users, movies, output_file)

        loss = calculate_rmse(result, labels)
        if return_loss:
            return loss
        else:
            print(f"RMSE for {self.__class__.__name__}: {loss}")
            return None
        
