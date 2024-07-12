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
    def __init__(self, num_users, num_items, embed_dim):
        super(GMF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

    def forward(self, users, items):
        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)
        return user_embeds * item_embeds

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, hidden_dims: list[int]):
        super(MLP, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(embed_dim * 2, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

    def forward(self, users, items):
        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)
        concat = torch.cat((user_embeds, item_embeds), dim=1)
        for i in range(len(self.fc_layers) - 1):
            concat = torch.relu(self.fc_layers[i](concat))
        return self.fc_layers[-1](concat) 


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_embed_dim, mlp_embed_dim, mlp_hidden_dims: list[int]):
        super(NeuMF, self).__init__()
        self.gmf = GMF(num_users, num_items, gmf_embed_dim)
        self.mlp = MLP(num_users, num_items, mlp_embed_dim, mlp_hidden_dims)
        self.out = nn.Linear(gmf_embed_dim + mlp_hidden_dims[-1], 1, bias=False)

    def forward(self, users, items):
        gmf_out = self.gmf(users, items)
        mlp_out = self.mlp(users, items)
        concat = torch.cat((gmf_out, mlp_out), dim=1)
        return self.out(concat).squeeze()

def nanstd(o, dim, keepdim=False):
    # Compute the mean along the specified dimension, ignoring NaNs
    mean = torch.nanmean(o, dim=dim, keepdim=True)
    
    # Compute the squared differences from the mean
    squared_diff = torch.pow(o - mean, 2)
    
    # Compute the mean of the squared differences, ignoring NaNs
    variance = torch.nanmean(squared_diff, dim=dim, keepdim=keepdim)
    
    # Take the square root to get the standard deviation
    std_dev = torch.sqrt(variance)
    
    return std_dev

class ItemNormalizer():
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

    def fit(self, users, items, ratings):
        data = torch.full((self.num_users, self.num_items), torch.nan, device=DEVICE)
        data[users, items] = ratings.float()
        mean = torch.nanmean(data, dim=0)
        std = nanstd(data, dim=0)
        self.mean = mean
        self.std = std

    def transform(self, users, items, ratings):
        ratings_normalized = (ratings - self.mean[items]) / self.std[items]
        return ratings_normalized

    def inverse_transform(self, users, items, ratings):
        ratings_denormalized = ratings * self.std[items] + self.mean[items]
        return ratings_denormalized

class UserNormalizer():
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

    def fit(self, users, items, ratings):
        data = torch.full((self.num_users, self.num_items), torch.nan, device=DEVICE)
        data[users, items] = ratings.float()
        mean = torch.nanmean(data, dim=1)
        std = nanstd(data, dim=1)
        self.mean = mean
        self.std = std

    def transform(self, users, items, ratings):
        ratings_normalized = (ratings - self.mean[users]) / self.std[users]
        return ratings_normalized

    def inverse_transform(self, users, items, ratings):
        ratings_denormalized = ratings * self.std[users] + self.mean[users]
        return ratings_denormalized


class NFC():
    def __init__(self, num_users, num_items, config):
        self.num_users = num_users
        self.num_items = num_items
        self.normalizer = ItemNormalizer(num_users, num_items)
        self.init_config(config)
        self.model = self.create_model()
        self.grads = None
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # create checkpoint dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def init_config(self, config):
        self.model_id = config.get("model_id", "NCF")
        self.checkpoint_dir = os.path.join("checkpoints", self.model_id)
        self.gmf_embed_dim = config.get("gmf_embed_dim", 4)
        self.mlp_embed_dim = config.get("mlp_embed_dim", 4)
        self.mlp_hidden_dims = config.get("mlp_hidden_dims", [16, 8, 4])
        self.test_split = config.get("test_split", 0.2)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 256)
        self.learning_rate = config.get("learning_rate", 0.001)
    
    def create_model(self):
        model = NeuMF(self.num_users, self.num_items, self.gmf_embed_dim, self.mlp_embed_dim, self.mlp_hidden_dims).to(DEVICE)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters:", num_params)
        return model

    def load_model(self):
        model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self):
        model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save(self.model.state_dict(), model_path)

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
        train_dataset = MovieDataset(users_train, items_train, ratings_train)
        test_dataset = MovieDataset(users_test, items_test, ratings_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
            
    def train(self):
        users, items, ratings = self.load_data("data/data_train.csv")
        self.normalizer.fit(users, items, ratings)
        ratings_normalized = self.normalizer.transform(users, items, ratings)
        train_loader, test_loader = self.create_loaders(users, items, ratings_normalized)
        self.evaluate(test_loader)
        best_rmse = float("inf")
        for epoch in range(self.n_epochs):
            self.train_epoch(train_loader)
            rmse = self.evaluate(test_loader)
            if rmse < best_rmse:
                best_rmse = rmse
                self.save_model()

    def evaluate(self, dataloader):
        if len(dataloader) == 0:
            return float("inf")
        self.model.eval()
        rmse = 0.0
        with torch.no_grad():
            for users, items, ratings in tqdm.tqdm(dataloader):
                pred = self.model(users, items)
                pred = self.normalizer.inverse_transform(users, items, pred)
                ratings = self.normalizer.inverse_transform(users, items, ratings)
                rmse += (pred - ratings).pow(2).sum().item()
        rmse /= len(dataloader.dataset)
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

    def create_submission(self):
        if not os.path.exists("submissions"):
            os.makedirs("submissions")

        submission_path = os.path.join("submissions", f"{self.model_id}.csv")
        users, items, _ = self.load_data("data/sampleSubmission.csv")
        submission_dataset = MovieDataset(users, items, torch.zeros_like(users, dtype=torch.float32, device=DEVICE))
        submission_loader = torch.utils.data.DataLoader(submission_dataset, batch_size=256, shuffle=False)
        preds = self.predict(submission_loader)
        df = pd.read_csv("data/sampleSubmission.csv")
        df["Prediction"] = preds
        df.to_csv(submission_path, index=False)

if __name__ == "__main__":
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    config = {
        "model_id": "NCF",
        "gmf_embed_dim": 4,
        "mlp_embed_dim": 4,
        "mlp_hidden_dims": [16, 8, 4],
        "test_split": 0.2,
        "n_epochs": 10,
        "batch_size": 256,
        "learning_rate": 0.001
    }

    ncf = NFC(10000, 1000, config)
    ncf.train()
    ncf.create_submission()
