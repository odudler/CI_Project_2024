import math
import torch
import torch.nn as nn

from ncf_utils import create_interaction_matrix, DEVICE
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1, include_ffn=True):
        super().__init__()
        self.include_ffn = include_ffn
        self.multihead_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(embed_size)
        if include_ffn:
            self.ff = nn.Sequential(
                nn.Linear(embed_size, ff_hidden_size),
                nn.ReLU(),
                nn.Linear(ff_hidden_size, embed_size),
            )
            self.layernorm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)
        if self.include_ffn:
            ff_output = self.ff(x)
            x = x + self.dropout(ff_output)
            x = self.layernorm2(x)
        return x

class GMF(nn.Module):
    def __init__(self, num_users, num_items, config):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.init_config(config)

        self.user_embed = nn.Embedding(num_users, self.embed_dim)
        self.item_embed = nn.Embedding(num_items, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)

        if self.use_interactions and self.separate_embeddings:
            self.item_adjacent_embed = nn.Linear(num_items, self.embed_dim, bias=False)
            self.user_adjacent_embed = nn.Linear(num_users, self.embed_dim, bias=False)

        if self.use_interactions and self.include_attention_layer:
            seq_len = 12 if self.interaction_type == "split_by_rating" else 4
            self.pe = PositionalEncoding(self.embed_dim, seq_len)
            self.attention_layers = nn.ModuleList()
            for _ in range(self.n_attention_layers):
                self.attention_layers.append(TransformerEncoderLayer(self.embed_dim, self.n_heads, self.ff_hidden_dim, include_ffn=self.include_ffn))

        # TODO: refactor, this is a mess
        num_embeds = 1
        if self.use_interactions:
            # adjacent x adjacent
            if self.interaction_type == "split_by_rating":
                num_embeds += 5
            else:
                num_embeds += 1

            # user x user adjacent, item x item adjacent
            if self.include_same_hadamards:
                if self.interaction_type == "split_by_rating":
                    num_embeds += 10
                else: 
                    num_embeds += 2

            # user x item adjacent, item x user adjacent
            if self.include_cross_hadamards:
                if self.interaction_type == "split_by_rating":
                    num_embeds += 10
                else: 
                    num_embeds += 2

        self.out_dim = num_embeds * self.embed_dim

    def init_config(self, config):
        self.embed_dim = config.get("embed_dim", 4)
        self.dropout = config.get("dropout", 0.1)

        self.use_interactions = config.get("use_interactions", False)
        # all below only matter if use_interactions is True
        self.interaction_type = config.get("interaction_type", "default") # default, weighted_by_rating, split_by_rating
        self.separate_embeddings = config.get("separate_embeddings", False)
        self.include_same_hadamards = config.get("include_same_hadamards", False)
        self.include_cross_hadamards = config.get("include_cross_hadamards", False)

        self.include_attention_layer = config.get("include_attention_layer", False)
        # all below only matter if include_attention_layer is True
        self.n_attention_layers = config.get("n_attention_layers", 1)
        self.n_heads = config.get("n_heads", 1)
        self.include_ffn = config.get("include_ffn", False)
        self.ff_hidden_dim = config.get("ff_hidden_dim", 16)

    def set_interaction_matrices(self, dataset):
        if not self.use_interactions:
            return
        self.user_interaction_matrix = create_interaction_matrix(self.num_users, self.num_items, dataset, "user", self.interaction_type)
        self.item_interaction_matrix =  create_interaction_matrix(self.num_users, self.num_items, dataset, "item", self.interaction_type)

    def get_extra_state(self):
        if not self.use_interactions:
            return {}
        return {
            "user_interaction_matrix": self.user_interaction_matrix,
            "item_interaction_matrix": self.item_interaction_matrix
        }

    def set_extra_state(self, state):
        if not self.use_interactions:
            return
        self.user_interaction_matrix = state["user_interaction_matrix"]
        self.item_interaction_matrix = state["item_interaction_matrix"]

    def forward(self, users, items):
        if self.use_interactions:
            assert hasattr(self, "user_interaction_matrix") and hasattr(self, "item_interaction_matrix")

        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)
        user_embeds = self.dropout(user_embeds)
        item_embeds = self.dropout(item_embeds)

        if not self.use_interactions:
            return user_embeds * item_embeds

        if self.interaction_type == "split_by_rating":
            item_interaction_matrix = self.item_interaction_matrix[:,:,items].permute(0, 2, 1)
            item_interaction_matrix[:,torch.arange(len(items), device=DEVICE), users] = 0.
            user_interaction_matrix = self.user_interaction_matrix[:,users,:]
            user_interaction_matrix[:,torch.arange(len(users), device=DEVICE), items] = 0.
        else:
            item_interaction_matrix = self.item_interaction_matrix[:,items].T
            item_interaction_matrix[torch.arange(len(items), device=DEVICE), users] = 0.
            user_interaction_matrix = self.user_interaction_matrix[users,:]
            user_interaction_matrix[torch.arange(len(users), device=DEVICE), items] = 0.

        if self.separate_embeddings:
            adjacent_user_embeds = self.user_adjacent_embed(item_interaction_matrix)
            adjacent_item_embeds = self.item_adjacent_embed(user_interaction_matrix)
            adjacent_user_embeds = self.dropout(adjacent_user_embeds)
            adjacent_item_embeds = self.dropout(adjacent_item_embeds)
        else:
            adjacent_user_embeds = item_interaction_matrix @ self.user_embed.weight
            adjacent_item_embeds = user_interaction_matrix @ self.item_embed.weight

        if self.include_attention_layer:
            if self.interaction_type == "split_by_rating":
                all_embeds = torch.cat((user_embeds.unsqueeze(0), item_embeds.unsqueeze(0), adjacent_user_embeds, adjacent_item_embeds), dim=0) # [12, len(users), embed_dim]
                all_embeds = self.pe(all_embeds)
                for layer in self.attention_layers:
                    all_embeds = layer(all_embeds)
                user_embeds = all_embeds[0]
                item_embeds = all_embeds[1]
                adjacent_user_embeds = all_embeds[2:7]
                adjacent_item_embeds = all_embeds[7:12]
            else:
                all_embeds = torch.cat((user_embeds.unsqueeze(0), item_embeds.unsqueeze(0), adjacent_user_embeds.unsqueeze(0), adjacent_item_embeds.unsqueeze(0)), dim=0)
                all_embeds = self.pe(all_embeds)
                for layer in self.attention_layers:
                    all_embeds = layer(all_embeds)
                user_embeds = all_embeds[0]
                item_embeds = all_embeds[1]
                adjacent_user_embeds = all_embeds[2]
                adjacent_item_embeds = all_embeds[3]

        hadamards = []
        user_item_hadamard = user_embeds * item_embeds
        user_adj_item_adj_hadamard = adjacent_user_embeds * adjacent_item_embeds

        if self.interaction_type == "split_by_rating":
            user_adj_item_adj_hadamard = user_adj_item_adj_hadamard.permute(1, 0, 2).reshape(len(users), -1)

        hadamards.append(user_item_hadamard)
        hadamards.append(user_adj_item_adj_hadamard)
        if self.include_same_hadamards:
            user_user_adj_hadamard = user_embeds * adjacent_user_embeds
            item_item_adj_hadamard = item_embeds * adjacent_item_embeds

            if self.interaction_type == "split_by_rating":
                user_user_adj_hadamard = user_user_adj_hadamard.permute(1, 0, 2).reshape(len(users), -1)
                item_item_adj_hadamard = item_item_adj_hadamard.permute(1, 0, 2).reshape(len(items), -1)

            hadamards.append(user_user_adj_hadamard)
            hadamards.append(item_item_adj_hadamard)
        if self.include_cross_hadamards:
            user_item_adj_hadamard = user_embeds * adjacent_item_embeds
            item_user_adj_hadamard = item_embeds * adjacent_user_embeds

            if self.interaction_type == "split_by_rating":
                user_item_adj_hadamard = user_item_adj_hadamard.permute(1, 0, 2).reshape(len(users), -1)
                item_user_adj_hadamard = item_user_adj_hadamard.permute(1, 0, 2).reshape(len(items), -1)

            hadamards.append(user_item_adj_hadamard)
            hadamards.append(item_user_adj_hadamard)

        return torch.cat(hadamards, dim=1)

class MLP(nn.Module):
    def __init__(self, num_users, num_items, config):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.init_config(config)

        self.user_embed = nn.Embedding(num_users, self.embed_dim)
        self.item_embed = nn.Embedding(num_items, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)

        if self.use_interactions and self.separate_embeddings:
            self.item_adjacent_embed = nn.Linear(num_items, self.embed_dim, bias=False)
            self.user_adjacent_embed = nn.Linear(num_users, self.embed_dim, bias=False)

        if self.use_interactions and self.include_attention_layer:
            seq_len = 12 if self.interaction_type == "split_by_rating" else 4
            self.pe = PositionalEncoding(self.embed_dim, seq_len)
            self.attention_layers = nn.ModuleList()
            for _ in range(self.n_attention_layers):
                self.attention_layers.append(TransformerEncoderLayer(self.embed_dim, self.n_heads, self.ff_hidden_dim, include_ffn=self.include_ffn))

        self.fc_layers = nn.ModuleList()
        num_embeds = 2
        if self.use_interactions:
            num_embeds += 2
        if self.interaction_type == "split_by_rating":
            num_embeds += 8

        self.fc_layers.append(nn.Linear(self.embed_dim * num_embeds, self.hidden_dims[0]))
        for i in range(1, len(self.hidden_dims)):
            self.fc_layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))

        self.out_dim = self.hidden_dims[-1]
    
    def init_config(self, config):
        self.embed_dim = config.get("embed_dim", 4)
        self.hidden_dims = config.get("hidden_dims", [16, 8, 4])
        self.dropout = config.get("dropout", 0.1)

        self.use_interactions = config.get("use_interactions", False)
        # all below only matter if use_interactions is True
        self.interaction_type = config.get("interaction_type", "default") # default, weighted_by_rating, split_by_rating
        self.separate_embeddings = config.get("separate_embeddings", False)

        self.include_attention_layer = config.get("include_attention_layer", False)
        # all below only matter if include_attention_layer is True
        self.n_attention_layers = config.get("n_attention_layers", 1)
        self.n_heads = config.get("n_heads", 1)
        self.include_ffn = config.get("include_ffn", False)
        self.ff_hidden_dim = config.get("ff_hidden_dim", 16)
    
    def set_interaction_matrices(self, dataset):
        if not self.use_interactions:
            return
        self.user_interaction_matrix = create_interaction_matrix(self.num_users, self.num_items, dataset, "user", self.interaction_type)
        self.item_interaction_matrix =  create_interaction_matrix(self.num_users, self.num_items, dataset, "item", self.interaction_type)

    def get_extra_state(self):
        if not self.use_interactions:
            return {}
        return {
            "user_interaction_matrix": self.user_interaction_matrix,
            "item_interaction_matrix": self.item_interaction_matrix
        }

    def set_extra_state(self, state):
        if not self.use_interactions:
            return
        self.user_interaction_matrix = state["user_interaction_matrix"]
        self.item_interaction_matrix = state["item_interaction_matrix"]

    def forward(self, users, items):
        if self.use_interactions:
            assert hasattr(self, "user_interaction_matrix") and hasattr(self, "item_interaction_matrix")

        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)
        user_embeds = self.dropout(user_embeds)
        item_embeds = self.dropout(item_embeds)
        concat = torch.cat((user_embeds, item_embeds), dim=1)

        if self.use_interactions:
            if self.interaction_type == "split_by_rating":
                item_interaction_matrix = self.item_interaction_matrix[:,:,items].permute(0, 2, 1)
                item_interaction_matrix[:,torch.arange(len(items), device=DEVICE), users] = 0.
                user_interaction_matrix = self.user_interaction_matrix[:,users,:]
                user_interaction_matrix[:,torch.arange(len(users), device=DEVICE), items] = 0.
            else:
                item_interaction_matrix = self.item_interaction_matrix[:,items].T
                item_interaction_matrix[torch.arange(len(items), device=DEVICE), users] = 0.
                user_interaction_matrix = self.user_interaction_matrix[users,:]
                user_interaction_matrix[torch.arange(len(users), device=DEVICE), items] = 0.

            if self.separate_embeddings:
                adjacent_user_embeds = self.user_adjacent_embed(item_interaction_matrix)
                adjacent_item_embeds = self.item_adjacent_embed(user_interaction_matrix)
                adjacent_user_embeds = self.dropout(adjacent_user_embeds)
                adjacent_item_embeds = self.dropout(adjacent_item_embeds)
            else:
                adjacent_user_embeds = item_interaction_matrix @ self.user_embed.weight
                adjacent_item_embeds = user_interaction_matrix @ self.item_embed.weight

            if self.include_attention_layer:
                if self.interaction_type == "split_by_rating":
                    all_embeds = torch.cat((user_embeds.unsqueeze(0), item_embeds.unsqueeze(0), adjacent_user_embeds, adjacent_item_embeds), dim=0) # [12, len(users), embed_dim]
                    all_embeds = self.pe(all_embeds)
                    for layer in self.attention_layers:
                        all_embeds = layer(all_embeds)
                    user_embeds = all_embeds[0]
                    item_embeds = all_embeds[1]
                    adjacent_user_embeds = all_embeds[2:7]
                    adjacent_item_embeds = all_embeds[7:12]
                else:
                    all_embeds = torch.cat((user_embeds.unsqueeze(0), item_embeds.unsqueeze(0), adjacent_user_embeds.unsqueeze(0), adjacent_item_embeds.unsqueeze(0)), dim=0)
                    all_embeds = self.pe(all_embeds)
                    for layer in self.attention_layers:
                        all_embeds = layer(all_embeds)
                    user_embeds = all_embeds[0]
                    item_embeds = all_embeds[1]
                    adjacent_user_embeds = all_embeds[2]
                    adjacent_item_embeds = all_embeds[3]


            if self.interaction_type == "split_by_rating":
                adjacent_user_embeds = adjacent_user_embeds.permute(1, 0, 2).reshape(len(users), -1)
                adjacent_item_embeds = adjacent_item_embeds.permute(1, 0, 2).reshape(len(items), -1)

            concat = torch.cat((concat, adjacent_user_embeds, adjacent_item_embeds), dim=1)

        for i in range(len(self.fc_layers) - 1):
            concat = torch.relu(self.fc_layers[i](concat))
        return self.fc_layers[-1](concat) 


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, config):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.init_config(config)

        if self.typ != "mlp":
            self.gmf = GMF(num_users, num_items, self.gmf_config)
        if self.typ != "gmf":
            self.mlp = MLP(num_users, num_items, self.mlp_config)

        if self.typ == "gmf":
            self.gmf_head = nn.Linear(self.gmf.out_dim, 1, bias=False)
        elif self.typ == "mlp":
            self.mlp_head = nn.Linear(self.mlp.out_dim, 1, bias=False)
        elif self.typ == "both":
            self.both_head = nn.Linear(self.gmf.out_dim + self.mlp.out_dim, 1, bias=False)
        else:
            raise ValueError("Invalid type")

    def init_config(self, config):
        self.typ = config.get("type", "both") # gmf, mlp, both
        self.gmf_config = config.get("gmf", {})
        self.mlp_config = config.get("mlp", {})

    def set_interaction_matrices(self, dataset):
        if self.typ != "mlp":
            self.gmf.set_interaction_matrices(dataset)
        if self.typ != "gmf":
            self.mlp.set_interaction_matrices(dataset)

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
