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

    def __init__(self, hidden_size, label_size, param_init):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.param_init = param_init

        # Layers for attention, comparison, aggregation
        self.attend = self._mlp(self.hidden_size, self.hidden_size)
        self.compare = self._mlp(2 * self.hidden_size, self.hidden_size)
        self.aggregate = self._mlp(2 * self.hidden_size, self.hidden_size)

        self.output = nn.Linear(self.hidden_size, self.label_size)
        self.log_prob = nn.LogSoftmax()

        # Inititiate parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.param_init)
                m.bias.data.normal_(0, self.param_init)

        def _mlp(self, in_dim, out_dim):
            nn_layers = []
            nn_layers.append(nn.Dropout(p=0.2))
            nn_layers.append(nn.Linear(in_dim, out_dim))
            nn_layers.append(nn.ReLU())
            nn_layers.append(nn.Dropout(p=0.2))
            nn_layers.append(nn.Linear(out_dim))
            nn_layers.append(nn.ReLU())
            return nn.Sequential(*nn_layers)

        def forward(self, s1_linear, s2_linear):
            """s1 := a, s2 := b in the paper"""
            len1 = s1_linear.size(1)
            len2 = s2_linear.size(1)

            # Attention
            f1 = self.attend(s1_linear.view(-1, self.hidden_size))
            f2 = self.attend(s2_linear.view(-1, self.hidden_size))
            # e_{ij}: batch_size * len_1 * len_2  namedtensor this
            attn_weights_1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
            attn_weights_2 = torch.transpose(attn_weights_1, 1, 2)

            # Normalize
            prob1 = F.softmax(attn_weights_1.view(-1, len2)
                              ).view(-1, len1, len2)
            prob2 = F.softmax(attn_weights_2.view(-1, len1)
                              ).view(-1, len2, len1)
            s1_weights = torch.cat((s1_linear, torch.bmm(prob1, s2_linear)), 2)
            s2_weights = torch.cat((s2_linear, torch.bmm(prob2, s1_linear)), 2)

            # Compare
            g1 = self.compare(s1_weights.view(-1, 2 * self.hidden_size))
            g1 = g1.view(-1, len1, self.hidden_size)
            g2 = self.compare(s2_weights.view(-1, 2 * self.hidden_size))
            g2 = g2.view(-1, len2, self.hidden_size)

            # Aggregate
            v1 = torch.squeeze(torch.sum(g1, 1), 1)
            v2 = torch.squeeze(torch.sum(g2, 1), 1)

            h = self.aggregate(torch.cat((v1, v2, 1)))

            return self.log_prob(self.output(h))
