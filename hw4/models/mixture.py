# Code for mixture of models trained with exact log marginal likelihood and VAE

import torch
from namedtensor import ntorch
ds = ntorch.distributions
from collections import defaultdict


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

    def __init__(self, q_inference, *models, kl_weight=0.5):
        super().__init__()
        self.q = q_inference
        self.kl_weight = kl_weight
        self.num_samples = num_samples

        self.models = ntorch.nn.ModuleList(models)
        self.K = len(self.models)

    def reinforce(self, premise, hypothesis, label):
        # REINFORCE
        q = self.q(premise, hypothesis, label).rename('label', 'latent')
        latent_dist = ds.Categorical(logits=q, dim_logit='latent')

        one_hot = torch.eye(4).index_select(0, label.values)
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
        prior = ds.Categorical(logits=ones, dim_logit='latent')

        KLD = ds.kl_divergence(latent_dist, prior) * self.kl_weight
        loss = kl.mean() - surrogate.mean()  # -(surrogate.mean() - kl.mean())
        return loss, loss.detach()

    def decode(self, premise, hypothesis):
        label = ntorch.ones(premise.shape['batch'], names=('batch',))
        predictions = 0
        for i in range(1, self.num_samples+1):
            q = self.q(premise, hypothesis, label *
                       i).rename('label', 'latent').exp()
            for c in range(len(self.models)):
                log_probs = self.models[c](premise, hypothesis)
                preds += log_probs * q.get('latent', c) / len(self.models)
        return predictions / self.num_samples

    def get_loss(self, premise, hypothesis, label):
        return self.reinforce(premise, hypothesis, label)

    def forward(self, premise, hypothesis, label):
        return self.infer(premise, hypothesis)
