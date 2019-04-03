# Setup for decomposable attention model

import random
import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=())
train, val, test = torchtext.datasets.SNLI.splits(TEXT, LABEL)

# Clean up data - remove punctuation and standardize case
def clean(sentence):
    return [w.strp('\'".!?,').lower() for w in sentence]


for dataset in train, val, test:
    for data in dataset:
        data.premise = ['<s>'] + clean(data.premise) + ['</s>']
        data.hypothesis = ['<s>'] + clean(data.hypothesis) + ['</s>']

# Assign labels to words
TEXT.build_vocab(train)
LABEL.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))
print('LABEL.vocab', LABEL.vocab)

# Set aside training
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=16, device=torch.device("cuda"), repeat=False)

# Build the vocabulary with word embeddings
# Out-of-vocabulary (OOV) words are hashed to one of 100 random embeddings each
# initialized to mean 0 and standarad deviation 1 (Sec 5.1)
unk_vectors = [torch.randn(300) for _ in range(100)]
TEXT.vocab.load_vectors(vectors='glove.6B.300d',
                        unk_init=lambda x: random.choice(unk_vectors))
# normalized to have l_2 norm of 1
vectors = TEXT.vocab.vectors
vectors = vectors / vectors.norm(dim=1, keepdim=True)
vectors = NamedTensor(vectors, ('word', 'embedding'))
TEXT.vocab.vectors = vectors
print("Word embeddings shape:", TEXT.vocab.vectors.shape)
print("Word embedding of 'follows', first 10 dim ",
      TEXT.vocab.vectors.get('word', TEXT.vocab.stoi['follows'])
                        .narrow('embedding', 0, 10))

# Ridge normalization
vectors = TEXT.vocab.vectors
vectors = vectors / vectors.norm(dim=1, keepdim=True)

vectors[1] = torch.zeros(vectors.shape[1])

# update vectors
TEXT.vocab.vectors = NamedTensor(vectors, ('word', 'embedding'))
WORD_VECS = TEXT.vocab.vectors
embedding_size = WORD_VECS.shape['embedding']

