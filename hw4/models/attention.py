# Basic and intra-attention implementation of decomposable attention model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, param_init):
        super(encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.param_init = param_init

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.input = nn.Linear(self.embedding_size, self.hidden_size)

    def forward(self, sent1, sent2):
        """Compare the sentences, preform NLI"""
        s1 = self.embedding(sent1)
        s2 = self.embedding(sent2)
        s1.view(-1, self.embedding_size)
        s2.view(-1, self.embedding_size)

        batch_size = sent1.size(0)
        s1_input = self.input(sent1).view(batch_size, -1, self.hidden_size)
        s2_input = self.input(sent2).view(batch_size, -1, self.hidden_size)
        return s1_input, s2_input


class AttentionModel(nn.Module):
    """
    Model as implemented in the paper
    - NNs for attend, compare, and aggregate
    """
    raise NotImplemented
