import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import numpy as np

from grokfast import gradfilter_ema
from normalizer import ItemNormalizer, UserNormalizer, BothNormalizer
from ncf_models import NeuMF
from ncf_utils import load_data, create_train_eval_split, DEVICE
from ncf_configs import ncf_config, ncf_extended_config, ncf_extended_attention_config


class NCFTrain():
    def __init__(self, num_users, num_items, config):
        self.num_users = num_users
        self.num_items = num_items
        self.init_config(config)

        # create checkpoint dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.normalizer = self.create_normalizer()
        self.model = self.create_model()
        self.grads = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

    def init_config(self, config):
        self.model_id = config.get("model_id", "NCF")
        self.save_model = config.get("save_model", False)
        self.verbose = config.get("verbose", False)
        self.checkpoint_dir = os.path.join("checkpoints", self.model_id)

        # NCF model config
        self.model_config = config.get("model_config", {})

        # Training config
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 256)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.use_lr_scheduler = config.get("use_lr_scheduler", False)
        self.use_grokfast = config.get("use_grokfast", False)
        self.weight_decay = config.get("weight_decay", 0.01)

        # Normalizer config
        self.normalizer = config.get("normalizer", "both") # item, user, both
        self.divide_by_std = config.get("divide_by_std", False)

    def create_normalizer(self):
        if self.normalizer == "item":
            return ItemNormalizer(self.num_users, self.num_items, self.divide_by_std)
        elif self.normalizer == "user":
            return UserNormalizer(self.num_users, self.num_items, self.divide_by_std)
        elif self.normalizer == "both":
            return BothNormalizer(self.num_users, self.num_items, self.divide_by_std)
        else:
            raise ValueError("Invalid normalizer")
    
    def create_model(self):
        model = NeuMF(self.num_users, self.num_items, self.model_config).to(DEVICE)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Created model with {num_params} parameters")
        return model

    def save(self, file_name):
        if not self.save_model:
            return
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict()
        }
        model_path = os.path.join(self.checkpoint_dir, f"{file_name}.pth")
        torch.save(state, model_path)

    def load(self, file_name, exclude_optimizer=False):
        model_path = os.path.join(self.checkpoint_dir, f"{file_name}.pth")
        state = torch.load(model_path)
        self.model.load_state_dict(state["model"])
        self.normalizer.load_state_dict(state["normalizer"])
        if not exclude_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        print("Model loaded")

    def train_epoch(self, dataloader):
        if len(dataloader) == 0:
            return float("inf")
        self.model.train()
        rmse = 0.0
        criterion = nn.MSELoss()
        it = tqdm.tqdm(dataloader) if self.verbose else dataloader
        for users, items, ratings in it:
            self.optimizer.zero_grad()
            pred = self.model(users, items)
            ratings_normalized = self.normalizer.transform(users, items, ratings)
            loss = criterion(pred, ratings_normalized)
            pred_denormalized = self.normalizer.inverse_transform(users, items, pred)
            rmse += (pred_denormalized - ratings).pow(2).sum().item()
            loss.backward()
            if self.use_grokfast:
                self.grads = gradfilter_ema(self.model, grads=self.grads)
            self.optimizer.step()
        rmse /= len(dataloader.dataset)
        rmse = np.sqrt(rmse)
        print("Train RMSE:", rmse)
        return rmse

    def train(self, train_dataset, eval_dataset=None):
        self.model.set_interaction_matrices(train_dataset)
        self.normalizer.fit(train_dataset.users, train_dataset.items, train_dataset.ratings)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_each_step = eval_dataset is not None

        best_rmse = float("inf")
        for epoch in range(self.n_epochs):
            print(f"\nEpoch {epoch + 1}/{self.n_epochs}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            rmse = self.train_epoch(train_loader)
            if eval_each_step:
                rmse = self.evaluate(eval_dataset)
            if self.use_lr_scheduler:
                self.scheduler.step(rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                self.save("best_model")
            self.save("last_model")

    def evaluate(self, dataset):
        if len(dataset) == 0:
            return float("inf")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        rmse = 0.0
        with torch.no_grad():
            it = tqdm.tqdm(dataloader) if self.verbose else dataloader
            for users, items, ratings in it:
                pred = self.model(users, items)
                pred_denormalized = self.normalizer.inverse_transform(users, items, pred)
                rmse += (pred_denormalized - ratings).pow(2).sum().item() / len(dataloader.dataset)
        rmse = np.sqrt(rmse)
        print("Eval RMSE:", rmse)
        return rmse

    def predict(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            it = tqdm.tqdm(dataloader) if self.verbose else dataloader
            for users, items, _ in it:
                pred = self.model(users, items)
                pred = self.normalizer.inverse_transform(users, items, pred)
                preds.append(pred)
        return torch.cat(preds).cpu().numpy()

    def create_submission(self, file_suffix=""):
        if not os.path.exists("submissions"):
            os.makedirs("submissions")

        submission_path = os.path.join("submissions", f"{self.model_id}{file_suffix}.csv")
        dataset = load_data("data/sampleSubmission.csv")
        preds = self.predict(dataset)
        df = pd.read_csv("data/sampleSubmission.csv")
        df["Prediction"] = preds
        df.to_csv(submission_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ncf") # ncf, ncf_extended, ncf_extended_attention
    args = parser.parse_args()

    config = {
        "ncf": ncf_config,
        "ncf_extended": ncf_extended_config,
        "ncf_extended_attention": ncf_extended_attention_config
    }[args.model]

    rmses = []
    for i in range(10):
        torch.manual_seed(i)
        np.random.seed(i)

        dataset = load_data("data/data_train.csv")
        train_dataset, eval_dataset = create_train_eval_split(dataset, train_split=0.9, seed=i)

        ncf = NCFTrain(10000, 1000, config=config)
        ncf.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
        ncf.load("best_model")
        rmse = ncf.evaluate(eval_dataset)
        rmses.append(rmse)
        ncf.create_submission(f"_{i}")

    print("Mean RMSE:", np.mean(rmses), np.std(rmses))
    print("RMSEs:", rmses)
