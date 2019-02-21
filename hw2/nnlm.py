from data_prep import torch, TEXT  

WORD_VECS = TEXT.vocab.vectors  

BATCH_SIZE = 10

class NNLM(torch.nn.Module):
    """Implement neural net convolutional language model inspired by Bengio et al. 2003"""  
    def __init__(self, n, hidden_size=100, num_layers=3, dropout_rate=0.3, num_filters=50):
        super().__init__()
      
        self.n = n
        self.vocab_size = len(TEXT.vocab)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.embedding = torch.nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)
        self.conv = torch.nn.Conv1d(300, num_filters, n-1, stride=1)
        
        # Setup layers  
        linear_blocks = [  # input
            torch.nn.Linear(num_filters, hidden_size),
            torch.nn.Tanh()]
        for l in range(num_layers - 2):  # hidden
            linear_blocks.append(torch.nn.Linear(num_filters, hidden_size))
            linear_blocks.append(torch.nn.Tanh)
        linear_blocks.extend([  # output
            torch.nn.Linear(hidden_size, self.vocab_size), 
            torch.nn.Softmax(dim=-1)])
        
        self.out = torch.nn.Sequential(*linear_blocks)
        
    def forward(self, x):
        x = x.transpose(0, 1)
        embeddings = self.dropout(self.embeddings(x)).permute(1, 2, 0)
        convs = self.conv(embeddings).transpose(1, 2)
        
        out = self.out(convs).flatten(0, 1)
        return out
      
    def format_batch(self, batch):
        list_samples = []
        list_results = []
        
        num_words = 0
        batch_size = len(batch)
        
        for i in range(batch_size):
            context = batch.text[{'batch': i}].values.data
            targets = batch.target.get('batch', i).values.data  
            
            num_words += len(sent)
            padded_context = torch.nn.functional.pad(context, (self.n - 2, batch_size - len(context)), value=0)
            target_words = torch.nn.functional.pad(targets, (0, batch_size - len(targets)), values=0)
            
            list_samples.append(padded_context)
            list_results.append(target_words)
            
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
            
        