import os
import torch
import torch.nn as nn
import pandas as pd
import tqdm
import numpy as np

from utils import extract_users_items_predictions
from grokfast import gradfilter_ema

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:",DEVICE)


class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class GMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, include_adjacent=False, user_interaction_matrix=None, item_interaction_matrix=None):
        super(GMF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.include_adjacent = include_adjacent
        if include_adjacent:
            assert user_interaction_matrix is not None and item_interaction_matrix is not None
            self.user_interaction_matrix = user_interaction_matrix
            self.item_interaction_matrix = item_interaction_matrix
        self.out_dim = 4 * embed_dim if include_adjacent else embed_dim

    def forward(self, users, items):
        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)
        if not self.include_adjacent:
            return user_embeds * item_embeds

        item_interaction_matrix = self.item_interaction_matrix[:,items]
        item_interaction_matrix[users, torch.arange(len(items))] = 0.
        user_interaction_matrix = self.user_interaction_matrix[users,:]
        user_interaction_matrix[torch.arange(len(users)), items] = 0.
        adjacent_user_embeds = item_interaction_matrix.T @ self.user_embed.weight
        adjacent_item_embeds = user_interaction_matrix @ self.item_embed.weight

        embeds = user_embeds * item_embeds
        adjacent_embeds = adjacent_user_embeds * adjacent_item_embeds
        user_adj_embeds = user_embeds * adjacent_user_embeds
        item_adj_embeds = item_embeds * adjacent_item_embeds
        return torch.cat((embeds, adjacent_embeds, user_adj_embeds, item_adj_embeds), dim=1)

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, hidden_dims: list[int], include_adjacent=False, user_interaction_matrix=None, item_interaction_matrix=None):
        super(MLP, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.fc_layers = nn.ModuleList()
        num_embeds = 4 if include_adjacent else 2
        self.fc_layers.append(nn.Linear(embed_dim * num_embeds, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        self.include_adjacent = include_adjacent
        if include_adjacent:
            assert user_interaction_matrix is not None and item_interaction_matrix is not None
            self.user_interaction_matrix = user_interaction_matrix
            self.item_interaction_matrix = item_interaction_matrix
        
        self.out_dim = hidden_dims[-1]

    def forward(self, users, items):
        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)

        if not self.include_adjacent:
            concat = torch.cat((user_embeds, item_embeds), dim=1)
            for i in range(len(self.fc_layers) - 1):
                concat = torch.relu(self.fc_layers[i](concat))
            return self.fc_layers[-1](concat)
 
        item_interaction_matrix = self.item_interaction_matrix[:,items]
        item_interaction_matrix[users, torch.arange(len(items))] = 0.
        user_interaction_matrix = self.user_interaction_matrix[users,:]
        user_interaction_matrix[torch.arange(len(users)), items] = 0.
        adjacent_user_embeds = item_interaction_matrix.T @ self.user_embed.weight
        adjacent_item_embeds = user_interaction_matrix @ self.item_embed.weight

        concat = torch.cat((user_embeds, item_embeds, adjacent_user_embeds, adjacent_item_embeds), dim=1)
        for i in range(len(self.fc_layers) - 1):
            concat = torch.relu(self.fc_layers[i](concat))
        return self.fc_layers[-1](concat) 


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_embed_dim, mlp_embed_dim, mlp_hidden_dims: list[int], include_adjacent=False, user_interaction_matrix=None, item_interaction_matrix=None, type="both"):
        super(NeuMF, self).__init__()
        self.typ = type
        if type != "mlp":
            self.gmf = GMF(num_users, num_items, gmf_embed_dim, include_adjacent, user_interaction_matrix, item_interaction_matrix)
        if type != "gmf":
            self.mlp = MLP(num_users, num_items, mlp_embed_dim, mlp_hidden_dims, include_adjacent, user_interaction_matrix, item_interaction_matrix)

        if type == "gmf":
            self.gmf_head = nn.Linear(self.gmf.out_dim, 1, bias=False)
        elif type == "mlp":
            self.mlp_head = nn.Linear(self.mlp.out_dim, 1, bias=False)
        elif type == "both":
            self.both_head = nn.Linear(self.gmf.out_dim + self.mlp.out_dim, 1, bias=False)
        else:
            raise ValueError("Invalid type")

    def forward(self, users, items):
        if self.typ == "gmf":
            gmf_out = self.gmf(users, items)
            return self.gmf_head(gmf_out).squeeze()
        elif self.typ == "mlp":
            mlp_out = self.mlp(users, items)
            return self.mlp_head(mlp_out).squeeze()
        elif self.typ == "both":
            gmf_out = self.gmf(users, items)
            mlp_out = self.mlp(users, items)
            concat = torch.cat((gmf_out, mlp_out), dim=1)
            out = self.both_head(concat)
            return out.squeeze()
        else:
            raise ValueError("Invalid type")

def nanstd(o, dim, keepdim=False):
    # Compute the mean along the specified dimension, ignoring NaNs
    mean = torch.nanmean(o, dim=dim, keepdim=True)
    
    # Compute the squared differences from the mean
    squared_diff = torch.pow(o - mean, 2)
    
    # Compute the mean of the squared differences, ignoring NaNs
    variance = torch.nanmean(squared_diff, dim=dim, keepdim=keepdim)
    
    # Take the square root to get the standard deviation
    std_dev = torch.sqrt(variance)

    mask = std_dev == 0.
    std_dev[mask] = 1.0

    return std_dev

class ItemNormalizer():
    def __init__(self, num_users, num_items, divide_by_std=False):
        self.num_users = num_users
        self.num_items = num_items
        self.divide_by_std = divide_by_std

    def fit(self, users, items, ratings):
        data = torch.full((self.num_users, self.num_items), torch.nan, device=DEVICE)
        data[users, items] = ratings.float()
        self.mean = torch.nanmean(data, dim=0)
        self.std = nanstd(data, dim=0)

    def transform(self, users, items, ratings):
        ratings_normalized = ratings - self.mean[items]
        if self.divide_by_std:
            ratings_normalized = ratings_normalized / self.std[items]
        return ratings_normalized

    def inverse_transform(self, users, items, ratings):
        if self.divide_by_std:
            ratings = ratings * self.std[items]
        ratings_denormalized = ratings + self.mean[items]
        return torch.clamp(ratings_denormalized, 1, 5)

    def save(self, file_path):
        torch.save({
            "mean": self.mean,
            "std": self.std
        }, file_path)

    def load(self, file_path):
        state_dict = torch.load(file_path)
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

class UserNormalizer():
    def __init__(self, num_users, num_items, divide_by_std=False):
        self.num_users = num_users
        self.num_items = num_items
        self.divide_by_std = divide_by_std

    def fit(self, users, items, ratings):
        data = torch.full((self.num_users, self.num_items), torch.nan, device=DEVICE)
        data[users, items] = ratings.float()
        self.mean = torch.nanmean(data, dim=1)
        self.std = nanstd(data, dim=1)

    def transform(self, users, items, ratings):
        ratings_normalized = ratings - self.mean[users]
        if self.divide_by_std:
            ratings_normalized = ratings_normalized / self.std[users]
        return ratings_normalized

    def inverse_transform(self, users, items, ratings):
        if self.divide_by_std:
            ratings = ratings * self.std[users]
        ratings_denormalized = ratings + self.mean[users]
        return torch.clamp(ratings_denormalized, 1, 5)
    
    def save(self, file_path):
        torch.save({
            "mean": self.mean,
            "std": self.std
        }, file_path)

    def load(self, file_path):
        state_dict = torch.load(file_path)
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

class BothNormalizer():
    def __init__(self, num_users, num_items, divide_by_std=False):
        self.num_users = num_users
        self.num_items = num_items
        self.divide_by_std = divide_by_std

    def fit(self, users, items, ratings):
        data = torch.full((self.num_users, self.num_items), torch.nan, device=DEVICE)
        data[users, items] = ratings.float()
        self.total_mean = torch.nanmean(data)
        data = data - self.total_mean
        self.user_mean = torch.nanmean(data, dim=1)
        self.item_mean = torch.nanmean(data, dim=0)

        self.user_std = nanstd(data, dim=1)
        self.item_std = nanstd(data, dim=0)

    def transform(self, users, items, ratings):
        ratings_normalized = ratings - self.total_mean - self.user_mean[users] - self.item_mean[items]
        if self.divide_by_std:
            ratings_normalized = ratings_normalized / (self.user_std[users] * self.item_std[items])
        return ratings_normalized

    def inverse_transform(self, users, items, ratings):
        if self.divide_by_std:
            ratings = ratings * (self.user_std[users] * self.item_std[items])
        ratings_denormalized = ratings + self.total_mean + self.user_mean[users] + self.item_mean[items]
        return torch.clamp(ratings_denormalized, 1, 5)

    def save(self, file_path):
        torch.save({
            "total_mean": self.total_mean,
            "user_mean": self.user_mean,
            "item_mean": self.item_mean,
            "user_std": self.user_std,
            "item_std": self.item_std
        }, file_path)
    
    def load(self, file_path):  
        state_dict = torch.load(file_path)
        self.total_mean = state_dict["total_mean"]
        self.user_mean = state_dict["user_mean"]
        self.item_mean = state_dict["item_mean"]
        self.user_std = state_dict["user_std"]
        self.item_std = state_dict["item_std"]

class NFC():
    def __init__(self, num_users, num_items, config):
        self.num_users = num_users
        self.num_items = num_items
        self.init_config(config)
        self.normalizer = self.create_normalizer()
        self.model = self.create_model()
        self.grads = None
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

        # create checkpoint dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def init_config(self, config):
        self.model_id = config.get("model_id", "NCF")
        self.type = config.get("type", "both")
        self.checkpoint_dir = os.path.join("checkpoints", self.model_id)
        self.gmf_embed_dim = config.get("gmf_embed_dim", 4)
        self.mlp_embed_dim = config.get("mlp_embed_dim", 4)
        self.mlp_hidden_dims = config.get("mlp_hidden_dims", [16, 8, 4])
        self.test_split = config.get("test_split", 0.2)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 256)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.normalizer = config.get("normalizer", "item")
        self.normalizer_divide_by_std = config.get("normalizer_divide_by_std", False)
        self.include_adjacent = config.get("include_adjacent", False)
        self.use_grokfast = config.get("use_grokfast", False)
        self.weight_decay = config.get("weight_decay", 0.01)

    def create_normalizer(self):
        if self.normalizer == "item":
            return ItemNormalizer(self.num_users, self.num_items, self.normalizer_divide_by_std)
        elif self.normalizer == "user":
            return UserNormalizer(self.num_users, self.num_items, self.normalizer_divide_by_std)
        elif self.normalizer == "both":
            return BothNormalizer(self.num_users, self.num_items, self.normalizer_divide_by_std)
        else:
            raise ValueError("Invalid normalizer")
    
    def create_model(self):
        model = NeuMF(self.num_users, self.num_items, self.gmf_embed_dim, self.mlp_embed_dim, self.mlp_hidden_dims, self.include_adjacent, self.create_user_interaction_matrix(), self.create_item_interaction_matrix(), self.type).to(DEVICE)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters:", num_params)
        return model

    def load_model(self, file_name):
        model_path = os.path.join(self.checkpoint_dir, f"{file_name}.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.normalizer.load(os.path.join(self.checkpoint_dir, "normalizer.pth"))
        optimizer_path = os.path.join(self.checkpoint_dir, f"{file_name}_optimizer.pth")
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        print("Model loaded")

    def save_model(self, file_name):
        model_path = os.path.join(self.checkpoint_dir, f"{file_name}.pth")
        torch.save(self.model.state_dict(), model_path)
        optimizer_path = os.path.join(self.checkpoint_dir, f"{file_name}_optimizer.pth")
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def train_epoch(self, dataloader):
        assert len(dataloader) != 0
        self.model.train()
        rmse = 0.0
        for users, items, ratings in tqdm.tqdm(dataloader):
            self.optimizer.zero_grad()
            pred = self.model(users, items)
            loss = self.criterion(pred, ratings)
            pred_denormalized = self.normalizer.transform(users, items, pred)
            ratings_denormalized = self.normalizer.transform(users, items, ratings)
            rmse += (pred_denormalized - ratings_denormalized).pow(2).sum().item()
            loss.backward()
            if self.use_grokfast:
                self.grads = gradfilter_ema(self.model, grads=self.grads)
            self.optimizer.step()
        rmse /= len(dataloader.dataset)
        rmse = np.sqrt(rmse)
        print("Train RMSE:", rmse)

    def load_data(self, file_path):
        df_train = pd.read_csv(file_path)
        users, items, ratings = extract_users_items_predictions(df_train)

        users = torch.tensor(users).long().to(DEVICE)
        items = torch.tensor(items).long().to(DEVICE)
        ratings = torch.tensor(ratings).long().to(DEVICE)
        return users, items, ratings

    def create_user_interaction_matrix(self):
        if not self.include_adjacent:
            return None
        users, items, ratings = self.load_data("data/data_train.csv")

        interaction_matrix = torch.zeros((self.num_users, self.num_items), device=DEVICE)
        interaction_matrix[users, items] = ratings.float()
        return interaction_matrix / torch.clip(interaction_matrix.sum(dim=1, keepdim=True), 1, np.inf).sqrt()

    def create_item_interaction_matrix(self):
        if not self.include_adjacent:
            return None
        users, items, ratings = self.load_data("data/data_train.csv")

        interaction_matrix = torch.zeros((self.num_users, self.num_items), device=DEVICE)
        interaction_matrix[users, items] = ratings.float()
        return interaction_matrix / torch.clip(interaction_matrix.sum(dim=0, keepdim=True), 1, np.inf).sqrt()

    def create_train_test_split(self, users, items, ratings):
        assert len(users) == len(items) == len(ratings)

        # Shuffle the data
        indices = torch.randperm(len(users))
        users = users[indices]
        items = items[indices]
        ratings = ratings[indices]

        train_samples = int(len(users) * (1 - self.test_split))
        users_train = users[:train_samples]
        items_train = items[:train_samples]
        ratings_train = ratings[:train_samples]
        users_test = users[train_samples:]
        items_test = items[train_samples:]
        ratings_test = ratings[train_samples:]
        return users_train, items_train, ratings_train, users_test, items_test, ratings_test

    def create_loaders(self, users, items, ratings):
        users_train, items_train, ratings_train, users_test, items_test, ratings_test = self.create_train_test_split(users, items, ratings)
        # only use training data to fit the normalizer
        self.normalizer.fit(users_train, items_train, ratings_train)
        self.normalizer.save(os.path.join(self.checkpoint_dir, "normalizer.pth"))
        ratings_train_normalized = self.normalizer.transform(users_train, items_train, ratings_train)
        ratings_test_normalized = self.normalizer.transform(users_test, items_test, ratings_test)
        train_dataset = MovieDataset(users_train, items_train, ratings_train_normalized)
        test_dataset = MovieDataset(users_test, items_test, ratings_test_normalized)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
            
    def train(self):
        users, items, ratings = self.load_data("data/data_train.csv")
        train_loader, test_loader = self.create_loaders(users, items, ratings)
        best_rmse = self.evaluate(test_loader)
        if np.isnan(best_rmse):
            raise ValueError("NaN RMSE")
        self.save_model("best_model")
        for epoch in range(self.n_epochs):
            print(f"\nEpoch {epoch + 1}/{self.n_epochs}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            self.train_epoch(train_loader)
            rmse = self.evaluate(test_loader)
            self.scheduler.step(rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                self.save_model("best_model")
            self.save_model("last_model")

    def evaluate(self, dataloader):
        if len(dataloader) == 0:
            return float("inf")
        self.model.eval()
        rmse = 0.0
        with torch.no_grad():
            for users, items, ratings in tqdm.tqdm(dataloader):
                pred = self.model(users, items)
                pred_normalized = self.normalizer.inverse_transform(users, items, pred)
                ratings_normalized = self.normalizer.inverse_transform(users, items, ratings)
                rmse += (pred_normalized - ratings_normalized).pow(2).sum().item() / len(dataloader.dataset)
        rmse = np.sqrt(rmse)
        print("Eval RMSE:", rmse)
        return rmse

    def predict(self, dataloader):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for users, items, _ in tqdm.tqdm(dataloader):
                pred = self.model(users, items)
                pred = self.normalizer.inverse_transform(users, items, pred)
                preds.append(pred)
        return torch.cat(preds).cpu().numpy()

    def create_submission(self, file_suffix=""):
        if not os.path.exists("submissions"):
            os.makedirs("submissions")

        submission_path = os.path.join("submissions", f"{self.model_id}{file_suffix}.csv")
        users, items, _ = self.load_data("data/sampleSubmission.csv")
        submission_dataset = MovieDataset(users, items, torch.zeros_like(users, dtype=torch.float32, device=DEVICE))
        submission_loader = torch.utils.data.DataLoader(submission_dataset, batch_size=256, shuffle=False)
        preds = self.predict(submission_loader)
        df = pd.read_csv("data/sampleSubmission.csv")
        df["Prediction"] = preds
        df.to_csv(submission_path, index=False)

if __name__ == "__main__":
    for i in range(10):
        print(f"Seed: {i}")
        torch.manual_seed(i)
        np.random.seed(i)

        config = {
            "model_id": f"NCF_ensemble_{i}",
            "type": "both", # gmf, mlp, both
            "gmf_embed_dim": 16,
            "mlp_embed_dim": 16,
            "mlp_hidden_dims": [64, 16, 4],
            "test_split": 0.1,
            "n_epochs": 30,
            "batch_size": 256,
            "learning_rate": 0.001,
            "normalizer": "both", # item, user, both
            "normalizer_divide_by_std": False,
            "include_adjacent": True,
            "use_grokfast": False,
            "weight_decay": 0.5
        }

        ncf = NFC(10000, 1000, config)
        ncf.train()
        ncf.load_model("best_model")
        ncf.create_submission("_best")
