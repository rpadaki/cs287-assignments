import torch
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, text, hidden_size=500, num_layers=2, 
                 embedding_size=1000, dropout=0.0, bidirectional=False, **kwargs):
        super(EncoderLSTM, self).__init__(text, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = len(text.vocab)
        self.embedding_dim = embedding_size
        self.embeddings = nn.Embeeding(self.vocab_size, self.embedding_dim)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=self.dropout, batch_first=True, 
                            bidirectional=self.bidirectional)
        
    def forward(self, input_, hidden):
        x = self.embeddings(input_)  # batch size, sentence len, embedding dim
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        return output, hidden  # batch, sentence len, hidden size * 1 or 2


class DecoderLSTM(EncoderLSTM):
    """Inherit same architecture as encoder, but reversed"""
    def __init__(text, hidden_size=500, num_layers=2, dropout=0.0,
                 bidirectional_encoder=False, context_size=1, **kwargs):
        super(DecoderLSTM, self).__init__(text, **kwargs)
        self.encoder_directions = 2 if bidirectional_encoder else 1
        self.context_size = context_size
        
        input_dim = context_len * self.num_layers * self.encoder_directions + 1
        self.output = nn.Linear(input_dim * self.hidden_size, self.vocab_size)
        
    def forward(self, input_, hidden, context):
        """:context: tuple (h, c) of hidden and cell states from t-1 of encoder"""
        x = self.embeddings(input_)
        x = F.relu(x)
        output, hidden = self.lstm(x, hidden)
        
        x = torch.cat(context[:self.context_size])
        batch_size = context.size(1)
        sentence_len = output.size(1)
        
        # Convert this to named version? rn [batch_size, 1, hidden_size * context_size]
        x = x.permute(1, 0, 2).contiguous().view(batch_size, 1, -1)
        x = x.expand(-1, sentence_length, -1)
        output = torch.cat((output, x), dim=2)
        
        output = self.output(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden
       