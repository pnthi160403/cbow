import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, dropout=0.5):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.context_size = context_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeds = torch.sum(self.embeddings(inputs), dim=1)
        embeds = self.dropout(embeds)

        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs
    
__all__ = ["CBOW"]