# Code for mixture of models trained with exact log marginal likelihood and VAE

import torch
from namedtensor import ntorch
from collections import defaultdict

nds = ntorch.distributions


class LatentMixtureModel(ntorch.nn.Module):
    """
    Produces marginal likelihood with uniform prior over all models
    - Experiment with k := number of models in the ensemble
    """

    def __init__(self, *models):
        super().__init__()
        self.models = ntorch.nn.ModuleList(models)

    def forward(self, p, h):
        # Collect predictions from all models, then average.
        # :p: is predictions, :h: is hypotheses
        preds = [model(p, h) for model in self.models]
        return sum(preds) / len(self.models)


class VAE(ntorch.nn.Module):
    """VAE to train ensemble mixture of models"""

    def __init__(self, q_inference, *models, num_samples, kl_weight=0.5,
                 elbo_method='reinforce'):
        super().__init__()
        self.q = q_inference
        self.kl_weight = kl_weight
        self.num_samples = num_samples

        self.models = ntorch.nn.ModuleList(models)
        self.K = len(self.models)
        self.elbo_method = elbo_method

    def reinforce(self, premise, hypothesis, label):
        # REINFORCE
        q = self.q(premise, hypothesis, label).rename('label', 'latent')
        latent_dist = nds.Categorical(logits=q, dim_logit='latent')
        # Sample to appromixate E[]
        samples = latent_dist.sample([self.num_samples], names=('samples',))

        # Batch premises and hypotheses
        batches = defaultdict(list)
        premise_n = premise.unbind('batch')
        hypothesis_n = hypothesis.unbind('batch')

        # Get some samples
        samples_n = samples.transpose('batch', 'samples').tolist()

        # Idea is to work with samples based on their sampled model
        for i, batch in enumerate(samples_n):
            p = premise_n[i]
            h = hypothesis_n[i]
            for sample in batch:
                batches[sample].append((i, p, h))

        # Can now evaluate sampled models with batching
        batch_size = premise.shape['batch']
        counts = [0] * batch_size
        res = [None] * (self.num_samples * batch_size)

        correct = label.tolist()
        for i, items in batches.items():
            # for item in items:
            #     batch_p = ntorch.
            batch_p = ntorch.stack([p for _, p, _ in items], 'batch')
            batch_h = ntorch.stack([h for _, _, h in items], 'batch')
            batch_i = [i for i, _m _ in items]

            # Evaluate model per batch, then update
            preds = self.models[i](batch_p, batch_h)
            for i, log_probs in zip(ids, preds.unbind('batch')):
                res[self.sample_size * i + counts[i]
                    ] = log_probs.values[correct[i]]
                counts[i] += 1

        # Finally average results for sample
        res = torch.stack(res, dim=0).reshape(batch_size, self.num_samples)
        res = ntorch.tensor(res, names=('batch', 'sample',))

        # Onward to estimating gradient + calculating loss
        surrogate = (latent_dist.log_prob(samples) *
                     res.detach() + res).mean('sample')
        prior = ntorch.ones(self.K, names='latent').log_softmax(dim='latent')
        prior = nds.Categorical(logits=prior, dim_logit='latent')
        KLD = nds.kl_divergence(latent_dist, prior) * self.kl_weight

        loss = (KLD - surrogate._tensor).mean()  # -(surrogate = kl)
        elbo = (KLD.detach() - res.detach().mean('sample')).mean()
        return loss, elbo

    def exact(self, premise, hypothesis, label):
        q = self.q(premise, hypothesis, label).rename('label', 'latent')
        latent_dist = nds.Categorical(logits=q, dim_logit='latent')

        one_hot = torch.eye(4, out = torch.cuda.FloatTensor()).index_select(0, label.values)
        one_hot = ntorch.tensor(one_hot, names=('batch', 'label'))

        # Calculate p(y | a, b, c) across all models K
        surrogate = 0
        q = q.exp()
        for c in range(len(self.models)):
            log_probs = self.models[c](premise, hypothesis)
            model_probs = q.get('latent', c)
            surrogate += (log_probs * one_hot).sum('label') * model_probs

        # KL regularization
        ones = ntorch.ones(self.K, names='latent').log_softmax(dim='latent')
        prior = nds.Categorical(logits=ones, dim_logit='latent')

        KLD = nds.kl_divergence(latent_dist, prior) * self.kl_weight
        loss = KLD.mean() - surrogate._tensor.mean()  # -(surrogate.mean() - kl.mean())
        return loss, loss.detach()

    def decode(self, premise, hypothesis):
        label = ntorch.ones(premise.shape['batch'], names=('batch',))
        preds = 0
        for i in range(1, self.num_samples+1):
            q = self.q(premise, hypothesis, label *
                       i).rename('label', 'latent').exp()
            q = self.q(premise, hypothesis, label *
                       i).rename('label', 'latent').exp()
            for c in range(len(self.models)):
                log_probs = self.models[c](premise, hypothesis)
                preds += log_probs * q.get('latent', c) / len(self.models)
        return preds / self.num_samples

    def get_loss(self, premise, hypothesis, label):
        if self.elbo_method == 'reinforce':
            return self.reinforce(premise, hypothesis, label)
        else:
            return self.exact(premise, hypothesis, label)

    def forward(self, premise, hypothesis, label=None):
        return self.decode(premise, hypothesis)
