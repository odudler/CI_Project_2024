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
from ncf_configs import ncf_config, ncf_extended_config, ncf_extended_attention_config, ncf_extended_same_config, ncf_extended_cross_config, ncf_extended_same_cross_config, ncf_extended_gmf_config, ncf_extended_mlp_config, ncf_extended_gmf_int_default_config, ncf_extended_gmf_int_weight_config, ncf_extended_mlp_int_weight_config, ncf_extended_mlp_int_split_config, ncf_extended_no_int_config


class NCFTrain():
    def __init__(self, num_users, num_items, config):
        '''
        Parameters:
            num_users (int): Number of users
            num_items (int): Number of items
            config (dict): Configuration dictionary
            config["model_id"] (str): Model ID, checkpoints will be saved in "checkpoints/{model_id}"
            config["save_model"] (bool): Whether to save the best and last model each epoch
            config["verbose"] (bool): Whether to print tqdm progress bars
            config["n_epochs"] (int): Number of epochs
            config["batch_size"] (int): Batch size
            config["learning_rate"] (float): Learning rate
            config["use_lr_scheduler"] (bool): Whether to use an adaptive learning rate scheduler that halves the learning rate when the validation RMSE does not improve for 2 epochs
            config["use_grokfast"] (bool): Whether to use Grokfast to amplify low-frequency gradients
            config["weight_decay"] (float): Weight decay for AdamW optimizer
            config["normalizer"] (str): Normalizer type, "item", "user" or "both", see section 2.B.3 of our report
            config["divide_by_std"] (bool): Whether to divide the ratings by the standard deviation during normalization, see section 2.B.3 of our report

            config["model_config"] (dict): Model configuration dictionary
            config["model_config"]["type"] (str): Model type, "gmf", "mlp" or "both", always "both" in our experiments, "gmf" and "mlp" implement the submodels as standalone models

            config["model_config"]["gmf"] (dict): GMF configuration dictionary
            config["model_config"]["gmf"]["embed_dim"] (int): Embedding dimension for GMF
            config["model_config"]["gmf"]["dropout"] (float): Dropout rate for GMF, applied after the embedding layer
            config["model_config"]["gmf"]["use_interactions"] (bool): Whether to use interactions in GMF, activates our Extended NCF model for the GMF submodel, see section 2.B of our report
            config["model_config"]["gmf"]["interaction_type"] (str): Interaction type for GMF, "default", "weight_by_rating" or "split_by_rating", see section 2.B of our report
            config["model_config"]["gmf"]["separate_embeddings"] (bool): Whether to share the parameters of the embeddings for users (items) and user interations (item interactions) in GMF, see section 2.B of our report
            config["model_config"]["gmf"]["include_same_hadamards"] (bool): Whether to include user * user_interactions (item * item_interactions) Hadamard products in GMF, see section 2.B.1 of our report
            config["model_config"]["gmf"]["include_cross_hadamards"] (bool): Whether to include user * item_interactions (item * user_interactions) Hadamard products in GMF, see section 2.B.1 of our report
            config["model_config"]["gmf"]["include_attention_layer"] (bool): Whether to include an attention layer in GMF, see section 2.B.2 of our report
            config["model_config"]["gmf"]["n_attention_layers"] (int): Number of attention layers in GMF
            config["model_config"]["gmf"]["n_heads"] (int): Number of heads in the attention layer in GMF
            config["model_config"]["gmf"]["include_ffn"] (bool): Whether to include a feed-forward network in GMF
            config["model_config"]["gmf"]["ff_hidden_dim"] (int): Hidden dimension of the feed-forward network in GMF

            config["model_config"]["mlp"] (dict): MLP configuration dictionary
            config["model_config"]["mlp"]["embed_dim"] (int): Embedding dimension for MLP
            config["model_config"]["mlp"]["dropout"] (float): Dropout rate for MLP, applied after the embedding layer
            config["model_config"]["mlp"]["hidden_dims"] (list): List of hidden dimensions for the MLP, the last element is the output dimension
            config["model_config"]["mlp"]["use_interactions"] (bool): Whether to use interactions in MLP, activates our Extended NCF model for the MLP submodel, see section 2.B of our report
            config["model_config"]["mlp"]["interaction_type"] (str): Interaction type for MLP, "default", "weight_by_rating" or "split_by_rating", see section 2.B of our report
            config["model_config"]["mlp"]["separate_embeddings"] (bool): Whether to share the parameters of the embeddings for users (items) and user interations (item interactions) in MLP, see section 2.B of our report
            config["model_config"]["mlp"]["include_attention_layer"] (bool): Whether to include an attention layer in MLP, see section 2.B.2 of our report
            config["model_config"]["mlp"]["n_attention_layers"] (int): Number of attention layers in MLP
            config["model_config"]["mlp"]["n_heads"] (int): Number of heads in the attention layer in MLP
            config["model_config"]["mlp"]["include_ffn"] (bool): Whether to include a feed-forward network in MLP
            config["model_config"]["mlp"]["ff_hidden_dim"] (int): Hidden dimension of the feed-forward network in MLP

        '''
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
    parser.add_argument("--model", type=str, default="ncf") # ncf, ncf_extended, ncf_extended_attention, ncf_extended_same, ncf_extended_cross, ncf_extended_same_cross, ncf_extended_gmf, ncf_extended_mlp, ncf_extended_gmf_int_default, ncf_extended_gmf_int_weight, ncf_extended_mlp_int_weight, ncf_extended_mlp_int_split, ncf_extended_no_int
    args = parser.parse_args()

    config = {
        "ncf": ncf_config,
        "ncf_extended": ncf_extended_config,
        "ncf_extended_attention": ncf_extended_attention_config,
        "ncf_extended_same": ncf_extended_same_config,
        "ncf_extended_cross": ncf_extended_cross_config,
        "ncf_extended_same_cross": ncf_extended_same_cross_config,
        "ncf_extended_gmf": ncf_extended_gmf_config,
        "ncf_extended_mlp": ncf_extended_mlp_config,
        "ncf_extended_gmf_int_default": ncf_extended_gmf_int_default_config,
        "ncf_extended_gmf_int_weight": ncf_extended_gmf_int_weight_config,
        "ncf_extended_mlp_int_weight": ncf_extended_mlp_int_weight_config,
        "ncf_extended_mlp_int_split": ncf_extended_mlp_int_split_config,
        "ncf_extended_no_int": ncf_extended_no_int_config
    }[args.model]

    rmses = []
    for i in range(5):
        torch.manual_seed(i)
        np.random.seed(i)

        dataset = load_data("data/data_train.csv")
        train_dataset, eval_dataset = create_train_eval_split(dataset, train_split=0.8, seed=i)

        ncf = NCFTrain(10000, 1000, config=config)
        ncf.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
        ncf.load("best_model")
        rmse = ncf.evaluate(eval_dataset)
        rmses.append(rmse)
        ncf.create_submission(f"_{i}")

    print("Mean RMSE:", np.mean(rmses), np.std(rmses))
    print("RMSEs:", rmses)
