import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable

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


class LogisticRegression(nn.Module):
    """Logistic Regression Model"""
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        # Linear: vocab size x class number
        self.linear = ntorch.nn.Linear(input_size, output_size, bias=True)
    def forward(self, x):
        x = self.linear(x)
        return x.sigmoid()  


# TRAINING FUNCTIONS  
def batch_binary_accuracy(preds, labels):
    """Return accuracy per batch"""
    return (preds == labels).sum("batch").item() / len(labels)  


encoding = ntorch.tensor(torch.eye(len(TEXT.vocab)), 
                         names=("vocab", "vocabIndex")).cuda()


def batch_text_to_input(batch_text, encoding):
    """Convert batch text to linear input"""
    oneHot = encoding.index_select("vocabIndex", batch_text)
    setofwords, _ = oneHot.max("seqlen")
    input_var = Variable(torch.FloatTensor(setofwords.cpu().numpy().T))
    input_var = setofwords.transpose('batch', 'vocab')
    return input_var
    

def train(model, train_iter, criterion, optimizer, batch_size, input_size):
    """Training Function"""
    model.train()   

    epoch_loss = 0
    loss_fn = criterion
    
    for batch in train_iter:
        input_var = batch_text_to_input(batch.text, encoding)
        preds = model(input_var)
        loss = preds.reduce2(batch.label, loss_fn, ("batch", "vocab"))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
      
    return epoch_loss / len(train_iter)


def validate(model, val_iter, criterion):
    """Validation function, called at the end of each epoch"""
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in val_iter:
            input_var = batch_text_to_input(batch.text, encoding)
            preds = model(input_var)
            preds = preds.max("vocab")[1]
            acc = batch_binary_accuracy(preds, batch.label)
            
            epoch_acc += acc
            
    return epoch_acc / len(val_iter)


# MODEL SETUP  
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = 2  
BATCH_SIZE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LogisticRegression(INPUT_DIM, OUTPUT_DIM)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(DEVICE)  

N_EPOCHS = 10

# TRAINING  
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iter, criterion, optimizer, 10, INPUT_DIM)
    val_acc = validate(model, val_iter, criterion)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | \
          Val. Acc: {val_acc*100:.2f}% |')
