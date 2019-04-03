# Basic and intra-attention implementation of decomposable attention model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from namedtensor import ntorch
from setup import WORD_VECS, embedding_size


class encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, param_init):
        super(encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.param_init = param_init

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.input = nn.Linear(self.embedding_size, self.hidden_size)

        # Bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.param_init)
                m.bias.data.uniform_(-0.01, 0.01)

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

    def __init__(self, hidden_size, label_size, dropout, param_init):
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
            nn_layers.append(nn.Dropout(p=dropout))
            nn_layers.append(nn.Linear(in_dim, out_dim))
            nn_layers.append(nn.ReLU())
            nn_layers.append(nn.Dropout(p=dropout))
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

# Named Stuff Here


class NamedLogSoftmax(ntorch.nn.Module):
    def __init__(self, dim):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.log_softmax(self.dim)


class NamedReLU(ntorch.nn.Module):
    def forward(self, x):
        return x.relu()


class MLP(ntorch.nn.Module):
    def __init__(self, input_dim, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        nn_layers = [ntorch.nn.Dropout(dropout),
                     ntorch.nn.Linear(input_size, hidden_size).spec(
                         input_dim, 'hidden'),
                     NamedReLU()
                     ]
        for i in range(num_layers - 1):
            nn_layers.append(ntorch.nn.Dropout(droupout))
            nn_layers.append(ntorch.nn.Linear(
                hidden_size, hidden_size).spec('hidden', 'hidden'))
            nn_layers.append(NamedRelU())
        self.nn = torch.nn.Sequential(**nn_layers)

    def forward(self, x):
        return self.nn(x)


class AttentionInput(ntorch.nn.Module):
    def __init__(self, num_layers=2, hidden_size=200, dropout=0.2, intra_attn=False):
        super().__init__()
        self.cap = 5
        self.intra_attn = intra_attn
        self.embeddings = ntorch.nn.Embedding.from_pretrained(
            WORD_VECS.values, freeze=True)

        if self.intra_attn:
            self.bias = torch.nn.Parameter(torch.ones(2 * self.cap + 1))
            self.intra_layers = MLP(
                embedding_size, hidden_size, num_layers, dropout=dropout)

    def align(self, s):
        intra_s = self.intra_layers(s)
        intra_s = intra_s.values.transpose(0, 1)  # formatting
        batch_seq = torch.bmm(intra_s, instra_s.transpose(1, 2))

        batches = batch_seq.shape[0]
        seqlen = batch_seq.shape[1]

        align_matrix = torch.tensor(
            [[(i - j) for j in range(seqlen)] for i in range(seqlen)])

        align_matrix = torch.clamp(align_matrix, -self.cap, self.cap)
        align_matrix = align_matrix.unsqueeze(0).expand(
            batches, seqlen, seqlen)  # batch * seqlen * seqlen

        align_matrix_b = self.bias[align_matrix + self.cap]
        weights = torch.softmax(align_matrix_b + batch_seq, dim=2)
        s_ = torch.matmul(weights, s.values.transpose(0, 1))
        s_ = ntorch.tensor(s, ('batch', 'seqlen', 'embedding'))

        return ntorch.cat([s, s_], 'embedding')

    def forward(self, s1, s2):  # Attend
        s1 = self.embeddings(s1)
        s2 = self.embeddings(s2)
        if self.intra_attn:
            s1 = self.align(s1)
            s2 = self.align(s2)
        return s1, s2


class NamedAttentionModel(ntorch.nn.Module):
    def __init__(self, num_layers=2, hidden_size=200, dropout=0.2, intra_attn=False):
        self.input = AttentionInput(
            num_layers, hidden_size, dropout, intra_attn)
        embedding_size = 300
        if intra_attn:
            embedding_size = 2 * embedding_size

        # Layers for attention, comparison, aggregation
        self.attend = MLP(embedding_size, hidden_size, num_layers, dropout)
        self.compare = MLP(2 * embedding_size, hidden_size,
                           num_layers, dropout)
        self.aggregate = MLP(2 * hidden_size, hidden_size,
                             num_layers, dropout, input_dim='hidden')

        self.output = nn.Sequential(ntorch.nn.Linear(
            hidden_size, 4).spec('hidden', 'label'), NamedLogSoftmax(dim='label'))

    def forward(self, sent1, sent2):
        """Notation straight from paper this time"""
        a_, b_ = self.input(sent1, sent2)

        # Attention
        F_a_ = self.attend(a_)
        F_b_ = self.attend(b_)

        e = F_a_.dot('hidden', F_b_)
        alpha = e.softmax(dim='seqlen').dot('seqlen', a_)
        beta = e.softmax(dim='seqlen').dot('seqlen', b_)

        # Comparison
        v1 = self.compare(ntorch.cat([a_, beta], 'embedding'))
        v2 = self.compare(ntorch.cat([b_, alpha], 'embedding'))

        # Aggregation
        v1 = v1.sum('seqlen')
        v2 = v2.sum('seqlen')
        y_hat = self.output(self.aggregate(ntorch.cat([v1, v2], 'hidden')))
        return y_hat
