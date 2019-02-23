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
  counts = get_counts_from_data(train_iter)
  probs = [defaultdict(int) for k in range(n)]
  for k in range(1, n):
    for key, value in counts[k].items():
      probs[k][key] = counts[k][key] / counts[k-1][key[:k]]
  for key, value in counts[0].items():
    probs[0][key] = value/sum(counts[0].values())
  return probs

def get_perplexity(probs, alphas, n=3):
  weighted_probs = defaultdict(int)
  total = 0.0
  for key, value in probs[n-1].items():
    a = sum([probs[k][key[n-k-1:]] * alphas[k] for k in range(n)])
    weighted_probs[key] = a
    total += np.log(a)
  return np.exp(-total/len(probs[n-1])), weighted_probs

def get_alphas(samples = 100, n=3):
  for s in range(samples):
    candidate = np.random.rand(n)
    yield tuple((candidate/sum(candidate)).tolist())

b_ppl = 30
b_wp = None
b_alpha = None
for alpha in list(get_alphas(10, 3)):
  ppl, wp = get_perplexity(probs, alpha, 3)
  if ppl < b_ppl:
    b_wp = wp
    b_ppl = ppl
    b_alpha = alpha

def predict(wp, input, n=3):
  ngram = tuple([TEXT.vocab.stoi[word] for word in input.split()[1-n:]])
  out_probs = defaultdict(int)
  for word in probs[0].keys():
    out_probs[word] = wp[ngram + word]
  sorted_out = sorted(out_probs.items(), key = lambda x : x[1])
  return([a for a in [TEXT.vocab.itos[sorted_out[-1-k][0][0]] for k in range(22)]
         if a != "<unk>" and a != "<eos>"][:20])

inputs = open("input.txt", "r").read().splitlines()
with open("output.txt", 'w') as out:
  out.write("id,word\n")
  for id, line in enumerate(inputs):
    out.write(str(id) + "," + " ".join(predict(b_wp, line[:-4])) + "\n")