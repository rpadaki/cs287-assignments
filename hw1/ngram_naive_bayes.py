import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from nltk.util import ngrams
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

# Fields for processing  
def tokenizer(text):
  words = [f.lower() for f in text.split(" ")]
  token = ["1" + i for i in words] +  ["2" + "".join(i) for i in ngrams(text.split(" "), 2)]
  #+ ["1" + i for i in text.split(" ")]
  return token

NGRAMS = NamedField(names=('ngramlen',), sequential = True, tokenize=tokenizer)
LABEL = NamedField(sequential=False, names=(), unk_token=None, dtype=torch.float)

# Load and split data into training sets  
train, val, test = torchtext.datasets.SST.splits(
    NGRAMS, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

# Build vocab
NGRAM.build_vocab(train, min_freq = 2)
LABEL.build_vocab(train)  

# Set up batches for model input  
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
  (train, val, test), batch_size=10, device=torch.device('cuda'))

def generate_ngram_naive_bayes_model(training_iter, alpha):
  labelCounts = ntorch.ones(len(LABEL.vocab), names=("class")).cuda() * 0
  vocabCounts = ntorch.tensor([alpha[f[0]] for f in NGRAMS.vocab.itos], names=("vocab",)).cuda() * ntorch.ones(len(LABEL.vocab), names=("class",)).cuda()
  classes = ntorch.tensor(torch.eye(len(LABEL.vocab)), names=("class", "classIndex")).cuda()
  encoding = ntorch.tensor(torch.eye(len(NGRAMS.vocab)), names=("vocab", "vocabIndex")).cuda()
  for batch in training_iter:
    oneHot = encoding.index_select("vocabIndex", batch.text)
    setofwords, _ = oneHot.max("ngramlen")
    classRep = classes.index_select("classIndex", batch.label.long())
    labelCounts += classRep.sum("batch")
    vocabCounts += setofwords.dot("batch", classRep)

  p = vocabCounts.get("class", 1)
  q = vocabCounts.get("class", 0)
  r = ((p*q.sum())/(q*p.sum())).log()
  # r= (p/q).log()
  weight = r
  b = (labelCounts.get("class", 1)/labelCounts.get("class", 0)).log()

  def naive_bayes(test_batch):
    oneHotTest = encoding.index_select("vocabIndex", test_batch.cuda())
    setofwords, _ = oneHotTest.max("seqlen")
    y = (weight.dot("vocab", setofwords) + b).sigmoid() 
    return (y - 0.5) * (ntorch.tensor([-1., 1.], names=("class")).cuda()) + 0.5  
  return naive_bayes

model = generate_naive_bayes_model(train_iter, alpha = {"1": 1., "2": 5., "3": 20., "<": 1.})
