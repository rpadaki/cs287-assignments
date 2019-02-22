from data_prep import torch, TEXT  


class LSTM(nn.Module):
    """
    LSTM implementation
    :hidden_dim: dimensions in hidden layer
    :dropout: 0.65 from (Zaremba, et al., 2015)
    :tie weights: extension in (Press & Wolf., 2016)
    """
    def __init__(self, hidden_dim=100, TEXT=TEXT, batch_size=10, dropout=0.65, tie_weights=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.tie_weights = tie_weights
        
        vocab_size, embedding_dim = TEXT.vocab.vectors.shape  
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, dropout=0.5, bidirectional=False)
        if tieweights:
            self.linear = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=embedding_dim),
                                        nn.Linear(in_features=embedding_dim, out_features=vocab_size))
            self.linear[1].weight = self.embeddings.weight
        else:
            self.linear = nn.Linear(in_features=self.hidden_dim, out_features=vocab_size)
        self.drop = nn.Dropout(dropout)
        
    def init_hidden(self):
        direction = 2 if self.lstm.bidirectional else 1
        h1 = Variable(torch.zeros(direction*self.lstm.num_layers, self.batch_size, self.hidden_dim)).cuda()
        h2 = Variable(torch.zeros(direction*self.lstm.num_layers, self.batch_size, self.hidden_dim)).cuda()
        return (h1, h2) 
    
    def detach_hidden(self, hidden):
        """Used to help with gradients"""
        return tuple([h.detach() for h in hidden])
      
    def forward(self, x, hidden, train=True):
        embed = self.embeddings(x)
        embed = self.drop(embedded) if train else embedded
        
        output, hidden_output = self.lstm(embed, hidden)
        output = output.view(-1, output.size(2))
        dropped = self.drop(output) if train else output
        
        decoded = self.linear(dropped)
        logits = F.log_softmax(decoded, dim=1)
        return logits, self.detach_hidden(hidden_output)
      
