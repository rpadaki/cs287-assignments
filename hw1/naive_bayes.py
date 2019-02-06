import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

# Fields for processing  
TEXT = NamedField(names=('seqlen',))
LABEL = NamedField(sequential=False, names=(), unk_token=None, dtype=torch.float)

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

def generate_naive_bayes_model(training_iter, alpha):
  labelCounts = ntorch.ones(len(LABEL.vocab), names=("class")).cuda() * 0
  vocabCounts = ntorch.ones(len(TEXT.vocab), len(LABEL.vocab), names=("vocab", "class")).cuda() * alpha
  classes = ntorch.tensor(torch.eye(len(LABEL.vocab)), names=("class", "classIndex")).cuda()
  encoding = ntorch.tensor(torch.eye(len(TEXT.vocab)), names=("vocab", "vocabIndex")).cuda()
  for batch in training_iter:
    oneHot = encoding.index_select("vocabIndex", batch.text)
    setofwords, _ = oneHot.max("seqlen")
    classRep = classes.index_select("classIndex", batch.label.long())
    labelCounts += classRep.sum("batch")
    vocabCounts += setofwords.dot("batch", classRep)
    
  p = vocabCounts.get("class", 1)
  q = vocabCounts.get("class", 0)
  r = ((p*q.sum())/(q*p.sum())).log()
  weight = r
  b = (labelCounts.get("class", 1)/labelCounts.get("class", 0)).log()
  def naive_bayes(test_batch):
    oneHotTest = encoding.index_select("vocabIndex", test_batch.cuda())
    setofwords, _ = oneHotTest.max("seqlen")
    y = ((weight.dot("vocab", setofwords) + b).sigmoid() - 0.5)
    return (y - 0.5) * (ntorch.tensor([-1, 1], names=("class")).cuda()) + 0.5  
  return naive_bayes

model = generate_naive_bayes_model(train_iter, 1)
