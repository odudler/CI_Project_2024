import argparse
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from ncf_train import NCFTrain
from ncf_utils import load_data, create_train_eval_split

def train(config):
    dataset = load_data("data/data_train.csv")
    train_dataset, eval_dataset = create_train_eval_split(dataset, train_split=0.9, seed=0)

    ncf = NCFTrain(10000, 1000, config)
    ncf.train(train_dataset=train_dataset)
    score = ncf.evaluate(eval_dataset)
    return {"score": score}

ncf_search_space = {
    "model_id": "NCF",
    "save_model": False,
    "verbose": False,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": tune.choice([4, 8, 16, 32, 64]),
            "dropout": 0.0,
            "use_interactions": False,
        },
        "mlp": {
            "embed_dim": tune.choice([4, 8, 16, 32, 64]),
            "dropout": 0.0,
            "hidden_dims": tune.choice([[64, 16, 4], [32, 8, 2], [32, 8], [16, 4], [16], [128, 32, 8, 2]]),
            "use_interactions": False,
        },
    },
    "n_epochs": 10,
    "batch_size": 256,
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "use_lr_scheduler": False,
    "use_grokfast": tune.choice([True, False]),
    "weight_decay": tune.loguniform(1e-2, 1e-0),
    "normalizer": tune.choice(["item", "user", "both"]),
    "normalizer_divide_by_std": tune.choice([True, False]),
}

ncf_extended_search_space = {
    "model_id": "NCF_extended",
    "save_model": False,
    "verbose": False,
    "model_config": {
        "type": "both", # gmf, mlp, both
        "gmf": {
            "embed_dim": tune.choice([8, 16, 32]),
            "dropout": tune.uniform(0.0, 0.5),
            "use_interactions": False,
            "interaction_type": tune.choice(["default", "weighted_by_rating", "split_by_rating"]),
            "separate_embeddings": tune.choice([True, False]),
            "include_same_hadamards": tune.choice([True, False]),
            "include_cross_hadamards": tune.choice([True, False]),
            "include_attention_layer": False,
        },
        "mlp": {
            "embed_dim": tune.choice([4, 8, 16, 32, 64]),
            "dropout": tune.uniform(0.0, 0.5),
            "hidden_dims": tune.choice([[64, 16, 4], [32, 8, 2], [32, 8], [16, 4]]),
            "use_interactions": False,
            "interaction_type": tune.choice(["default", "weighted_by_rating", "split_by_rating"]),
            "separate_embeddings": tune.choice([True, False]),
            "include_attention_layer": False,
        },
    },
    "n_epochs": 10,
    "batch_size": 256,
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "use_lr_scheduler": False,
    "use_grokfast": tune.choice([True, False]),
    "weight_decay": tune.loguniform(1e-2, 1e-0),
    "normalizer": tune.choice(["item", "user", "both"]),
    "normalizer_divide_by_std": tune.choice([True, False]),
}

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_space", type=str, default="ncf") # ncf, ncf_extended
    args = parser.parse_args()

    ray.init(num_cpus=4)

    optuna_search = OptunaSearch(metric="score", mode="min")
    optuna_search = ConcurrencyLimiter(optuna_search, max_concurrent=4)

    analysis = tune.run(
        train,
        metric="score",
        mode="min",
        config=ncf_search_space if args.search_space == "ncf" else ncf_extended_search_space,
        search_alg=optuna_search,
        num_samples=100,
    )

    print("Best config: ", analysis.get_best_config(metric="score", mode="min"))
    print("Best score: ", analysis.get_best_trial(metric="score", mode="min").last_result["score"])
    analysis.dataframe().sort_values("score").to_csv(f"results_{args.search_space}.csv", index=False)
