from data_prep import torch, TEXT  

WORD_VECS = TEXT.vocab.vectors  

class RNN(nn.Module):
    """
    RNN Model for LSTMs and GRUs
    :rnn_type: should be 'LSTM' or 'GRU'
    :n_input: embedding dimensions (e.g. 300)  
    :dropout: 0.65 from (Zaremba, et al., 2015)
    :tie weights: extension in (Press & Wolf., 2016)
    :vocab_size: len(TEXT.vocab)
    """
    def __init__(self, rnn_type, n_input, n_hidden, n_layers, vocab_size, dropout=0.65, tie_weights=False, bidirectional=True):  
       
        super(RNNModel, self).__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)
        self.rnn = getattr(torch.nn, rnn_type)(n_input, n_hidden, num_layers=n_layers, 
            dropout=dropout, bidirectional=bidirectional)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(n_hidden, vocab_size), 
                                           torch.nn.Softmax(dim=-1))
        
        if tie_weights:  # Optionally tie weights? 
            if n_hidden != n_input:
                raise ValueError('When tying flags, number of hidden layers must be equal to embedding size')
            self.decoder.weight = self.encoder.weight
            
        self.init_weights()
        self.rnn_type = rnn_type
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_input = n_input

        self.recurrent = True
        
    def init_weights(self):
        init_range = 0.05
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, input, hidden):
        input = input.transpose(0, 1)
        emb = self.drop(self.encoder(input))
        output, (hidden, _) = self.rnn(emb)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                    weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
        else:
            return weight.new_zeros(self.n_layers, batch_size, self.n_hidden)

    def format_batch(self, batch):
        list_samples = []
        list_results = []
        
        n_words = 0
        batch_size = len(batch)
        
        for i in range(batch_size):
            context = batch.text[{'batch': i}].values.data
            targets = batch.target.get('batch', i).values.data  
            
            num_words += len(sent)
            padded_context = torch.nn.functional.pad(context, (self.n - 1, 0), value=0)
            padded_context_all = torch.stack([padded_context[i:i+ self.n - 1] for i in range(len(context))])
            
            list_samples.append(padded_context_all)
            list_results.append(context)
            
        samples_all = torch.stack(list_samples)
        results_all = torch.cat(list_results)
        
        return (samples_all, results_all, num_words)


    def get_data(self, batch_iter):
        list_batch_x = []
        list_batch_y = []
        
        total_words = 0
        
        for batch in batch_iter:
            x, y, num_words = self.format_batch(batch)
            x_batches.append(x)
            y_batches.append(y)
            total_words += num_words

        return list_batch_x, list_batch_y, total_words
      
