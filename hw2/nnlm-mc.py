from data_prep import torch, TEXT  

class NNLMMultiChannel(nn.Module):
    """
    Language Model inspired by Bengio et al. 2003, 
    using different embeddings layers as seen in Kim et al, 2014
    """
    def __init__(self, hidden_dim=100, TEXT = TEXT, dropout=0.5):
        super(NNLM, self).__init__()
        
        vocab_size, embedding_dim = TEXT.vocab.vectors.shape
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        self.embeddings_static = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_static.weight.data.copy_(TEXT.vocab.vectors)
        self.embeddings_static.weight.requires_grad = False
        
        self.input = nn.Linear(2*embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.TEXT = TEXT
        self.dropout = dropout
        
    def forward(self, x):
        embed = self.embeddings(x)
        embed_static = self.embeddings_static(x)
        embed_dropped = self.dropout(embed)
        embed_static_dropped = self.dropout(embed_static)
        x = torch.cat([embed_dropped, embed_static_dropped], dim=2)
        x = x.view(len(x), -1)
        x = F.tanh(self.input(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)
      
      
    def predict(self, x):
        """Predict things"""
        embed = self.embeddings(x)
        embed = torch.cat([self.embeddings(x), self.embeddings_static(x)], dim=1)
        embed = embedded.view(-1, 1).transpose(0, 1)
        activated = nn.Tanh(self.input(embed))
        logits = F.log_softmax(output, dim = 1)
        out_ids = np.argsort(logits.data[0].tolist())[-20:][::-1]
        out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
        return out_words
            
        
            
        