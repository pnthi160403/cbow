import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tokenizers import Tokenizer
from .model import CBOW
import torch.nn as nn
from bs4 import BeautifulSoup
import re

# clean data
def clean_data(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = (text.lower()).replace(" '", "'")
    return handle_special_char(text)

def handle_special_char(sent):
    sent = re.sub(r'([.,!?;(){}\[\]])', r' \1 ', sent)
    sent = re.sub(r'\s{2,}', ' ', sent)
    sent = sent.strip()
    return sent

# read dataset
def read_dataset(config):
    train_data_path = config["DATA"]["train_path"]
    val_data_path = config["DATA"]["val_path"]
    test_data_path = config["DATA"]["test_path"]

    train_data = load_dataset("csv", data_files=train_data_path)["train"]["text"]
    val_data = load_dataset("csv", data_files=val_data_path)["train"]["text"]
    test_data = load_dataset("csv", data_files=test_data_path)["train"]["text"]

    return train_data, val_data, test_data

# read tokenizer
def read_tokenizer(config):
    tokenizer_path = config['DATA']['tokenizer_path']
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

# handle word with context size
def handle_word_with_context_size(token_ids, context_size):
    targets = []
    contexts = []

    for i in range(context_size, len(token_ids) - context_size):
        target_word = [token_ids[i]]
        context_words = [token_ids[j] for j in range(i - context_size, i)] + \
                        [token_ids[j] for j in range(i + 1, i + context_size + 1)]
        targets.append(target_word)
        contexts.append(context_words)

    return targets, contexts

# Custom Dataset
# define custom dataset
class Word2VecDataset(Dataset):
    def __init__(self, dataset, tokenizer, context_size):
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.targets = []
        self.contexts = []
        for text in dataset:
            token_ids = self.tokenizer.encode(text).ids
            target, context = handle_word_with_context_size(token_ids, self.context_size)
            self.targets += target
            self.contexts += context

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        target = torch.tensor(self.targets[idx])
        context = torch.tensor(self.contexts[idx])
        return target, context
    
# get data loader
def get_dataloader(config, tokenizer, dataset, batch_size, shuffle):
    context_size = config["MODEL"]["context_size"]
    dataset = Word2VecDataset(dataset, tokenizer, context_size)
    dataset_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_dataloader

# get Adam optimizer
def get_Adam_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

# get nll loss
def get_nll_loss():
    return nn.NLLLoss()

# get model
def get_model_cbow(config, tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    context_size = config["MODEL"]["context_size"]
    embedding_dim = config["MODEL"]["embedding_dim"]
    dropout = config["MODEL"]["dropout"]
    device = config["TRAIN"]["device"]

    model = CBOW(vocab_size, embedding_dim, context_size, dropout)

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return model.to(device)

__all__ = [
    "read_dataset",
    "read_tokenizer",
    "get_dataloader",
    "get_model_cbow"
]