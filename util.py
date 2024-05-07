from torchmetrics import Accuracy
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# draw loss plot
def draw_loss_plot(config, losses):
    save_path = config["LOG"]["path"]
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.savefig(f"{save_path}/loss_plot.png")
    plt.close()

def calc_accuracy(preds, target, num_classes, device):
    accuracy = Accuracy(task="multiclass", average='weighted', num_classes=num_classes).to(device)
    return accuracy(preds, target)