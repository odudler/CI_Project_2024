# Full config for showing which hyperparameters the model has
full_config = {
    "model_id": "includes_all_hyperparameters",
    "save_model": True,
    "verbose": True,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": 32,
            "dropout": 0.3,
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
            "dropout": 0.1,
            "hidden_dims": [16, 4],
            "use_interactions": True,
            "interaction_type": "default",
            "separate_embeddings": True,
            "include_attention_layer": False,
            "n_attention_layers": 1,
            "n_heads": 1,
            "include_ffn": True,
            "ff_hidden_dim": 16,
        },
    },
    "n_epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.00003,
    "use_lr_scheduler": False,
    "use_grokfast": False,
    "weight_decay": 0.25,
    "normalizer": "both", # item, user, both
    "normalizer_divide_by_std": True,
}

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