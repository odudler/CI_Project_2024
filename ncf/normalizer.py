import torch

from ncf_utils import DEVICE

def nanstd(tensor, dim, keepdim=False):
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    squared_diff = (tensor - mean) ** 2
    var = torch.nanmean(squared_diff, dim=dim, keepdim=keepdim)
    return var ** 0.5

class ItemNormalizer():
    def __init__(self, num_users, num_items, divide_by_std=False):
        self.num_users = num_users
        self.num_items = num_items
        self.divide_by_std = divide_by_std

    def fit(self, users, items, ratings):
        data = torch.full((self.num_users, self.num_items), torch.nan, device=DEVICE)
        data[users, items] = ratings.float()
        self.mean = torch.nanmean(data, dim=0)
        std = nanstd(data, dim=0)
        mask = std == 0.
        self.std = std.masked_fill(mask, 1.)

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

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std
        }

    def load_state_dict(self, state_dict):
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
        std = nanstd(data, dim=1)
        mask = std == 0.
        self.std = std.masked_fill(mask, 1.)

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

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std
        }
    
    def load_state_dict(self, state_dict):
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
            divider = self.user_std[users] + self.item_std[items]
            mask = divider == 0.
            divider = divider.masked_fill(mask, 1.)
            ratings_normalized = ratings_normalized / divider
        return ratings_normalized

    def inverse_transform(self, users, items, ratings):
        if self.divide_by_std:
            divider = self.user_std[users] + self.item_std[items]
            mask = divider == 0.
            divider = divider.masked_fill(mask, 1.)
            ratings = ratings * divider
        ratings_denormalized = ratings + self.total_mean + self.user_mean[users] + self.item_mean[items]
        return torch.clamp(ratings_denormalized, 1, 5)

    def state_dict(self):
        return {
            "total_mean": self.total_mean,
            "user_mean": self.user_mean,
            "item_mean": self.item_mean,
            "user_std": self.user_std,
            "item_std": self.item_std
        }
    
    def load_state_dict(self, state_dict):
        self.total_mean = state_dict["total_mean"]
        self.user_mean = state_dict["user_mean"]
        self.item_mean = state_dict["item_mean"]
        self.user_std = state_dict["user_std"]
        self.item_std = state_dict["item_std"]