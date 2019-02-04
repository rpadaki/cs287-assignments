def generate_naive_bayes_model(training_iter, alpha):
  labelCounts = ntorch.ones(len(LABEL.vocab), names=("class")).cuda() * 0
  vocabCounts = ntorch.ones(len(TEXT.vocab), len(LABEL.vocab), names=("vocab", "class")).cuda()) * alpha
  classes = ntorch.tensor(torch.eye(len(LABEL.vocab)), names=("class", "classIndex")).cuda()
  encoding = ntorch.tensor(torch.eye(len(TEXT.vocab)), names=("vocab", "vocabIndex")).cuda()
  for batch in training_iter:
    oneHot = encoding.index_select("vocabIndex", batch.text)
    setofwords, _ = oneHot.max("seqlen")
    classRep = classes.index_select("classIndex", batch.label)
    labelCounts += classRep.sum("batch")
    vocabCounts += setofwords.dot("batch", classRep)
    
  p = vocabCounts.get("class", 1)
  q = vocabCounts.get("class", 0)
  r = ((p*q.sum())/(q*p.sum())).log()
  weight = r
  b = np.log(labelCounts.get("class", 1)/labelCounts.get("class", 0))
  def naive_bayes(test_batch):
    oneHotTest = encoding.index_select("vocabIndex", test_batch.cuda())
    setofwords, _ = oneHotTest.max("seqlen")
    y = ((weight.dot("vocab", setofwords) + b).sigmoid() - 0.5)
    return (y - 0.5) * (ntorch.tensor([-1, 1], names=("class")).cuda()) + 0.5  
  return naive_bayes