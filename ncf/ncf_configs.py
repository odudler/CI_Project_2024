# Configuration dicts for all our experiments with NCF models. Hyperparameters were found using ncf_hyperparam.py.

# Config for default NCF model
ncf_config = {
    "model_id": "ncf",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.0,
            "use_interactions": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.0,
            "hidden_dims": [128, 32, 8, 2],
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.19,
    "normalizer": "item", # item, user, both
    "normalizer_divide_by_std": True,
}

# Config for the Extended NCF model
ncf_extended_config = {
    "model_id": "ncf_extended",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF with Attention model
ncf_extended_attention_config = {
    "model_id": "ncf_extended_attention",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.283,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": True,
            "n_attention_layers": 2,
            "n_heads": 2,
            "include_ffn": True,
            "ff_hidden_dim": 64,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with same hadamards
ncf_extended_same_config = {
    "model_id": "ncf_extended_same",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": True,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with cross hadamards
ncf_extended_cross_config = {
    "model_id": "ncf_extended_cross",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": True,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with same and cross hadamards
ncf_extended_same_cross_config = {
    "model_id": "ncf_extended_same_cross",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": True,
            "include_cross_hadamards": True,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with just GMF
ncf_extended_gmf_config = {
    "model_id": "ncf_extended_gmf",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "gmf", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with just MLP
ncf_extended_mlp_config = {
    "model_id": "ncf_extended_mlp",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "mlp", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with gmf default interaction type
ncf_extended_gmf_int_default_config = {
    "model_id": "ncf_extended_gmf_int_default",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with gmf weight by rating interaction type
ncf_extended_gmf_int_weight_config = {
    "model_id": "ncf_extended_gmf_int_weight",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "weighted_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with mlp weight by rating interaction type
ncf_extended_mlp_int_weight_config = {
    "model_id": "ncf_extended_mlp_int_weight",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "weighted_by_rating",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model with mlp split by rating interaction type
ncf_extended_mlp_int_split_config = {
    "model_id": "ncf_extended_mlp_int_split",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": False,
            "include_same_hadamards": False,
            "include_cross_hadamards": False,
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "split_by_rating",
            "separate_embeddings": True,
            "include_attention_layer": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}

# Config for the Extended NCF model without interactions
ncf_extended_no_int_config = {
    "model_id": "ncf_extended_no_int",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.28,
            "use_interactions": False,
        },
        "mlp": {
            "embed_dim": 16,
            "dropout": 0.08,
            "hidden_dims": [16, 4],
            "use_interactions": False,
        },
    },
    "n_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.001,
    "use_lr_scheduler": True,
    "use_grokfast": False,
    "weight_decay": 0.26,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": False,
}