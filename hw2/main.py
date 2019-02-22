import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

import math
import numpy as np
import tqdm  # track that progress  

# Batching
from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset

# NamedTensor  
from namedtensor import ntorch
from namedtensor.text import NamedField

# Model Imports  
# from ngram import NGram  
from nnlm import NNLM  
from lstm import RNN 


def repackage_hidden(h):
    """Detach hidden state tensors from history w/ new batch"""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h) 


def train_model(model, train_iter, criterion, optimizer, lr=0.0001, weight_decay=0): 
    """Method to train model. Define optimizer = ADAM, Adagrad, etc (e.g. optim.Adam); criterion = torch.nn.NLLLoss()""" 
    x_train, y_train, n_train = model.get_data(train_iter)

    model.train()
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    if model.recurrent:
        hidden = model.init_hidden(BATCH_SIZE)

    for ix in range(len(x_train)):
        optimizer.zero_grad()
        x_batch = x_train[ix]; y_batch = y_train[ix]

        if model.recurrent:
            hidden = repackage_hidden(hidden)
            output, hidden = model.forward(x_batch, hidden)
            loss = criterion(output.view(-1, VOCAB_SIZE).log(), y_batch)
            loss.backward()
            # Helps prevent the exploding gradient problem in RNNs / LSTMs
            torch.nn.utils.clip_grad_norm(model.paramters(), 0.25)  
            for p in model.parameters():
                p.data.add_(-lr, )
        else:
            probs = model.forward(x_batch)
            loss = criterion(probs.log(), y_batch)
            loss.backward()
        optimizer.step()

    return validate_model(model, x_train, y_train, n_train)


def validate_model(model, x, y, n_words):
    """Method to evaluate model"""
    model.eval()

    if model.recurrent:
        hidden = model.init_hidden(BATCH_SIZE)

    with torch.no_grad():
        criterion = torch.nn.NLLLoss(reduction='sum')
        epoch_loss = 0
        n_correct = 0

        for i in range(len(x)):
            if model.recurrent:
                probs, hidden = model.forward(x[i], hidden)
                probs = probs.view(-1, VOCAB_SIZE)
                hidden = repackage_hidden(hidden)
            else:
                probs = model.forward(x[i])
            epoch_loss += criterion(probs.log(), y[i])
            n_correct += (probs.argmax(dim=1) == y[i]).sum().float()
        loss = epoch_loss / n_words
        return 100. * n_correct / n_words, loss, math.exp(loss)  # perplexity  


def train_val_model(model, n_epochs, criterion, optimizer):  
    """Wrapper to call training and evaluation"""  
    for epoch in range(n_epochs):
        train_acc, train_loss, train_ppl = train_model(model, train_iter, criterion, optimizer, 0.0001, 0)
        x_val, y_val, n_val = model.get_data(val_iter)
        val_acc, val_loss, val_ppl = validate_model(model, x_val, y_val, n_val)
        print(f'| Epoch: {epoch+1:02} | Train Acc: {train_acc*100:.2f}% | \
            Train Loss: {train_loss:.3f} | Train Ppl: {train_ppl:.3f} | \
            Val. Acc: {val_acc*100:.2f}% | Val. Loss: {val_loss:.3f} | Val. Ppl: {val_ppl:.3f}')


BATCH_SIZE = 10  
VOCAB_SIZE = len(TEXT.vocab)

# Example Training  
# Presets taken from Zaremba, Sutskever, Vinyals. 2015
vocab_size = len(TEXT.vocab)
model_type = 'LSTM'
input_size = 300  # embedding dimensions
num_hidden_per_layer = 200  
num_layers = 2
dropout = 0.65
tied_args = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RNNModel(model_type, vocab_size, input_size, num_hidden_per_layer, num_layers, dropout, tied_args).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1)

# Think I'm calculating perplexity incorrectly  
N_EPOCHS = 30

vocab_size = len(TEXT.vocab)

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iter, criterion, optimizer, vocab_size)
    val_acc = validate(model, val_iter, criterion)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | \
          Val. Acc: {val_acc*100:.2f}% |')







    
