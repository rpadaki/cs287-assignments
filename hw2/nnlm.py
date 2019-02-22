from data_prep import torch, TEXT  

class NNLM(nn.Module):
    """Language Model inspired by Bengio et al. 2003"""
    def __init__(self, hidden_dim=100, TEXT = TEXT):
        super(NNLM, self).__init__()
        
        vocab_size, embedding_dim = TEXT.vocab.vectors.shape
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.input = nn.Linear(2*embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.TEXT = TEXT
        
    def forward(self, x):
        x = self.embeddings(x)
        x = x.view(len(x), -1)
        x = F.tanh(self.input(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)
      
      
    def predict(self, x):
        """Predict things"""
        embed = self.embeddings(x)
        embed = embedded.view(-1, 1).transpose(0, 1)
        activated = nn.Tanh(self.input(embed))
        logits = F.log_softmax(output, dim = 1)
        out_ids = np.argsort(logits.data[0].tolist())[-20:][::-1]
        out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
        return out_words
            
        
            
        