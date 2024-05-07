import configparser
import os
import torch

def get_config():
    config = configparser.ConfigParser()

    config["CONFIG"] = {
        "path": "./config_epoch_{}.ini",
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
        "val_path": "../dataset/data/val.csv",
        "test_path": "../dataset/data/test.csv",
        "ratio_val": 0.1,
        "ratio_test": 0.02,
    }

    config["LOG"] = {
        "path": "./logs",
    }

    config["CHECKPOINT"] = {
        "path": "./checkpoints",
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
        config.write(f)

def load_config(config=None, epoch=0):
    if config:
        last_epoch_path = config["CONFIG"]["path"].format(config["TRAIN"]["epochs"])
    else:
        last_epoch_path = "./config_epoch_{}.ini".format(epoch)

    if not os.path.exists(last_epoch_path):
        print(f"Config file {last_epoch_path} not found")
        return None
    config = configparser.ConfigParser()
    config.read(last_epoch_path)
    return config

__all__ = ["get_config", "create_all_dirs", "save_config", "load_config"]