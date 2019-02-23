from collections import defaultdict

def kgrams_from_batch(batch, k):
  combined_batch = [i for j in batch.text.transpose("batch", "seqlen").tolist() for i in j]
  for x in range(len(combined_batch)-k+1):
    yield tuple(combined_batch[x:x+k])

def get_counts_from_data(train_iter, n=3):
  counts = [defaultdict(int) for k in range(n)]
  # counts[k] is the k+1-gram dictionary
  for batch in iter(train_iter):
    for k in range(n):
      for kgram in kgrams_from_batch(batch, k+1):
        counts[k][kgram] += 1
  return counts

def get_probs_from_counts(train_iter, n=3):
  counts = get_counts_from_data(train_iter, n)
  probs = [defaultdict(int) for k in range(n)]
  for k in range(1, n):
    for key, value in counts[k].items():
      probs[k][key] = counts[k][key] / counts[k-1][key[:k]]
  for key, value in counts[0].items():
    probs[0][key] = value/sum(counts[0].values())
  return probs

def get_perplexity(probs, alphas, n=3, val = val_iter):
  weighted_probs = defaultdict(int)
  total = 0.0
  count = 0
  for batch in val_iter:
    for key in kgrams_from_batch(batch, n):
      total += np.log(sum([probs[k][key[n-k-1:]] * alphas[k] for k in range(n)]))
      count += 1
  return np.exp(-total/count)

def get_alphas(samples = 100, n=3):
  for s in range(samples):
    candidate = np.random.rand(n)
    yield tuple((candidate/sum(candidate)).tolist())

def optimize(probs, n=3, samples=100):
  b_ppl = 1000
  b_alpha = None
  for alpha in list(get_alphas(samples, n)):
    ppl = get_perplexity(probs, alpha, n)
    if ppl < b_ppl:
      b_ppl = ppl
      b_alpha = alpha
  return b_ppl, b_alpha

def predict(probs, alpha, input, n=3):
  kgram = tuple([TEXT.vocab.stoi[word] for word in input.split()[1-n:]])
  out_probs = defaultdict(int)
  for word in probs[0].keys():
    ngram = kgram + word
    out_probs[word] = sum([probs[k][ngram[n-k-1:]] * alpha[k] for k in range(n)])
  sorted_out = sorted(out_probs.items(), key = lambda x : x[1])
  return([a for a in [TEXT.vocab.itos[sorted_out[-1-k][0][0]] for k in range(22)]
         if a != "<unk>" and a != "<eos>"][:20])

def kaggle(probs, b_alpha, n=3):
  inputs = open("input.txt", "r").read().splitlines()
  with open("output.txt", 'w') as out:
    out.write("id,word\n")
    for id, line in enumerate(inputs):
      out.write(str(id) + "," + " ".join(predict(probs, b_alpha, line[:-4], n)) + "\n")