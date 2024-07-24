import os
import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
from typing import Literal, Union

from utils import extract_users_items_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MovieDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def load_data(file_path):
    global_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(global_path, file_path)
    df_train = pd.read_csv(file_path)
    users, items, ratings = extract_users_items_predictions(df_train)

    users = torch.tensor(users).long().to(DEVICE)
    items = torch.tensor(items).long().to(DEVICE)
    ratings = torch.tensor(ratings).long().to(DEVICE)
    return MovieDataset(users, items, ratings)

def create_train_eval_split(dataset, train_split=0.8, seed=42):
    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size], generator=generator)

    train_dataset = MovieDataset(dataset.users[train_dataset.indices], dataset.items[train_dataset.indices], dataset.ratings[train_dataset.indices])
    eval_dataset = MovieDataset(dataset.users[eval_dataset.indices], dataset.items[eval_dataset.indices], dataset.ratings[eval_dataset.indices])
    return train_dataset, eval_dataset

def _create_interaction_matrix(num_users, num_items, dataset, user_or_item: Literal["user", "item"], type: Literal["default", "weighted"], rating: Union[int, None] = None):
    users = dataset.users
    items = dataset.items
    ratings = dataset.ratings
    if rating is not None:
        users = users[ratings == rating]
        items = items[ratings == rating]

    interaction_matrix = torch.zeros((num_users, num_items), device=DEVICE)
    interaction_matrix[users, items] = ratings.float() if type == "weighted" else 1.
    dim = 1 if user_or_item == "user" else 0
    return interaction_matrix / torch.clip(interaction_matrix.sum(dim=dim, keepdim=True), 1, np.inf).sqrt()

def create_interaction_matrix(num_users, num_items, dataset, user_or_item: Literal["user", "item"], type: Literal["default", "weighted_by_rating", "split_by_rating"]):
    if type == "default":
        return _create_interaction_matrix(num_users, num_items, dataset, user_or_item, "default")
    elif type == "weighted_by_rating":
        return _create_interaction_matrix(num_users, num_items, dataset, user_or_item, "weighted", rating=None)
    elif type == "split_by_rating":
        return torch.stack([_create_interaction_matrix(num_users, num_items, dataset, user_or_item, "default", rating) for rating in range(1, 6)])
    else:
        raise ValueError("Invalid type")
