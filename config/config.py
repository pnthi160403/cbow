import os
import torch
import json

def get_config():
    config = {}

    config["BASE_DIR"] = {
        "path": "./",
    }

    config["CONFIG"] = {
        "path": "config",
        "pattern": "config_epoch_{}.json",
    }

    config["MODEL"] = {
        "embedding_dim": 512,
        "context_size": 2,
        "dropout": 0.01,
    }

    config["TRAIN"] = {
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    config["DATA"] = {
        "train_path": "../dataset/data/vi.csv",
        "val_path": "../dataset/data/vi.csv",
        "test_path": "../dataset/data/vi.csv",
        "tokenizer_path": "./dataset/tokenizer.json",
    }

    config["LOG"] = {
        "path": "logs",
    }

    config["CHECKPOINT"] = {
        "path": "checkpoints",
        "model_name": "model.pt",
    }

    return config

def join_path(config):
    for section in ["LOG", "CHECKPOINT", "CONFIG"]:
        if config[section]["path"].startswith(config["BASE_DIR"]["path"]):
            continue
        config[section]["path"] = config["BASE_DIR"]["path"] + config[section]["path"]
    return config

def create_dirs(config):
    for section in ["LOG", "CHECKPOINT", "CONFIG"]:
        path = config[section]["path"]
        if not os.path.exists(path):
            os.makedirs(path)

def save_config(config, epoch):
    path = config["CONFIG"]["path"] + "/" + config["CONFIG"]["pattern"].format(epoch)
    print(f"Saving config to {path}")

    with open(path, "w") as f:
        json.dump(config, f, indent=4)

__all__ = ["get_config", "create_dirs", "save_config", "join_path"] 