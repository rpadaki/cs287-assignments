# Code for mixture of models trained with exact log marginal likelihood and VAE

import torch
from namedtensor import ntorch


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


class MixtureModel(ntorch.nn.Module):
    def __init__(self, *models, fine_tune=False):
        super().__init__()

        self.fine_tune = fine_tune
        if fine_tune:
            self.models = ntorch.nn.ModuleList(models)
        else:
            self.models = [m.eval() for m in models]
            self.cache = {}
        self.weights = torch.nn.Parameter(torch.rand(len(self.models)))

    def f(self, model, premise, hypothesis):
        key = (model, premise.values, hypothesis.values)
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = model(premise, hypothesis)
        return self.cache[key]

    def forward(self, premise, hypothesis):
        if self.fine_tune:
            preds = [model(premise, hypothesis) for model in self.models]
        else:
            with torch.no_grad():
                preds = [self.f(model, premise, hypothesis)
                         for model in self.models]

        softmax = self.weights.softmax(dim=0)
        return sum(pred * softmax[c] for c, pred in enumerate(preds))
