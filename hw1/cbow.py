import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

# Fields for processing  
TEXT = NamedField(names=('seqlen',))
LABEL = NamedField(sequential=False, names=(), unk_token=None)

# Load and split data into training sets  
train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

# Build vocab
TEXT.build_vocab(train)
LABEL.build_vocab(train) 

# Set up batches for model input  
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
  (train, val, test), batch_size=10, device=torch.device('cuda'))

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) 

# MODEL  
class CBOW(nn.Module):  
    """
    Continuous Bag of Words Model - inspired by 
    Tensor Considered Harmful Pt 2 example
    """
    def __init__(self, input_size, num_classes, batch_size, 
                 vocab_size=TEXT.vocab.vectors.size()[0],
                 embedding_dim=TEXT.vocab.vectors.size()[1],
                 pretrained_embeddings=TEXT.vocab.vectors):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.linear = nn.Linear(input_size, num_classes, bias=True)
        
    def forward(self, x):
        x = x.augment(self.embedding, 'h').transpose('h', 'seqlen')
        avg_embedding = x.mean('seqlen')
        out = avg_embedding.op(self.linear, classes='h').softmax('classes')
        return out  

# TRAINING FUNCTIONS
def batch_binary_accuracy(preds, labels):
    """Return accuracy per batch"""
    return (preds == labels).sum("batch").item() / len(labels)  


def train(model, train_iter, criterion, optimizer):
    """Training Function"""
    model.train()
    
    epoch_loss = 0
    
    loss_fn = criterion  
    
    for batch in train_iter:
        optimizer.zero_grad()
        preds = model(batch.text)
        loss = preds.reduce2(batch.label, loss_fn, ('batch', 'classes'))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(train_iter)
  

def validate(model, val_iter):
    """Validation Function"""
    
    model.eval()
    
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in val_iter:
            preds = model(batch.text)
            preds = preds.max("classes")[1]
            acc = batch_binary_accuracy(preds, batch.label)
            
            epoch_acc += acc
            
    return epoch_acc / len(val_iter)
    

# MODEL SETUP  
INPUT_DIM = 300
OUTPUT_DIM = 2  
BATCH_SIZE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CBOW(INPUT_DIM, OUTPUT_DIM, BATCH_SIZE)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(DEVICE)  

# TRAINING  
N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iter, criterion, optimizer)
    val_acc = validate(model, val_iter)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | \
          Val. Acc: {val_acc*100:.2f}% |')
