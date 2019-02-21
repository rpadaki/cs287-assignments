import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

import math
import numpy as np

# Batching
from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset

# NamedTensor  
from namedtensor import ntorch
from namedtensor.text import NamedField

# Set default to cuda tensor with GPU
torch.set_default_tensor_type(torch.cuda.FloatTensor)  


# Batching Class for RNNs    
class NamedBpttIterator(BPTTIterator):
    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device)
        data = (data
            .stack(("seqlen", "batch"), "flat")
            .split("flat", ("batch", "seqlen"), batch=self.batch_size)
            .transpose("seqlen", "batch")
        )

        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text = data.narrow("seqlen", i, seq_len),
                    target = data.narrow("seqlen", i+1, seq_len),
                )
                         
            if not self.repeat:
                return


# Our input $x$
TEXT = NamedField(names=("seqlen",))

GLOVE = False  # Use GloVe embeddings
DEBUG_MODE = False  # When debugging use smaller vocab size 

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

if GLOVE:
    TEXT.build_vocab(train, vectors='glove.6B.100d')
else:
    TEXT.build_vocab(train)

    # Build vocab with embeddings  
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

if DEBUG_MODE:
    TEXT.build_vocab(train, max_size=1000)
else:
    TEXT.build_vocab(train)

# Split up data  
train_iter, val_iter, test_iter = NamedBpttIterator.splits(
    (train, val, test), batch_size=10, device=torch.device("cuda"), bptt_len=32, repeat=False)