import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from surprise import SVDpp
from utils import *

class SVDplusplus:
    def __init__(self, args):
        self.n_factors = args.svdpp.n_factors
        self.lr_all = args.svdpp.lr_all
        self.reg_all = args.svdpp.reg_all
        self.n_epochs = args.svdpp.n_epochs
        self.verbose = args.verbose
        self.random_state = args.random_state
        
    def train(self, df):
        dataset = prepare_data_for_surprise(df)
        trainset = dataset.build_full_trainset()
        svdpp = SVDpp(
            n_factors = self.n_factors,
            lr_all = self.lr_all,
            reg_all = self.reg_all,
            n_epochs = self.n_epochs,
            verbose = self.verbose,
            random_state = self.random_state
        )
        svdpp.fit(trainset)
        self.model = svdpp

    def predict(self, df, output_file = None, return_loss=False):
        users, movies, labels = extract_users_items_predictions(df)
        predictions = np.empty(len(labels))
        
        for i in tqdm(range(len(users)), desc = "Prediction Loop"):
            predictions[i] = self.model.predict(users[i], movies[i], verbose=self.verbose).est

        if output_file is not None:
            create_submission_from_array(predictions, users, movies, output_file)
        
        loss = calculate_rmse(predictions, labels)
        if return_loss:
            return loss
        else:
            print(f"RMSE for {self.__class__.__name__}: {loss}")


