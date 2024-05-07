import os
import torch
import json

def get_config():
    config = {}

    config["BASE_DIR"] = {
        "path": "./",
    }

    config["CONFIG"] = {
        "path": config["BASE_DIR"]["path"] + "config_epoch_{}.json",
    }

    config["MODEL"] = {
        "embedding_dim": 512,
        "context_size": 2,
        "dropout": 0.5,
    }

    config["TRAIN"] = {
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    config["DATA"] = {
        "train_path": "../dataset/data/vi.csv",
        "tokenizer_path": "./dataset/tokenizer.json",
        "ratio_val": 0.1,
        "ratio_test": 0.02,
    }

    config["LOG"] = {
        "path": config["BASE_DIR"]["path"] + "logs",
    }

    config["CHECKPOINT"] = {
        "path": config["BASE_DIR"]["path"] + "checkpoints",
    }

    return config

def create_all_dirs(config):
    for section in ["LOG", "CHECKPOINT"]:
        path = config[section]["path"]
        if not os.path.exists(path):
            os.makedirs(path)

def save_config(config, epoch):
    path = config["DEFAULT"]["path"].format(epoch)
    print(f"Saving config to {path}")

    with open(path, "w") as f:
        json.dump(config, f, indent=4)

__all__ = ["get_config", "create_all_dirs", "save_config"]