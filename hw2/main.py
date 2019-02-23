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
from nnlm import NNLM  
from lstm import LSTM  

# Training and Submission Methods   
from utils import train_val_model, train_val_model_lstm, write_to_kaggle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstm_model = LSTM()
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, lstm_model.parameters()), lr = 1e-3)
train_val_model_lstm(lstm_model, 5, opt)









    
