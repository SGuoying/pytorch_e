# %%
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class HiddenBayesianNet(pl.LightningModule):
    def __init__(self, layers: nn.ModuleList, vocab_size: int, hidden_dim: int, learning_rate: float, 
                    has_pre_layernorm: bool=False, has_post_layernorm: bool=False,
                    to_non_negative: Callable=torch.exp, sharing_weight: bool=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.digup = nn.Linear(hidden_dim, vocab_size, bias=False)
        if has_pre_layernorm:
            self.pre_layernorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        if has_post_layernorm:
            self.post_layernorm = nn.LayerNorm(vocab_size, eps=1e-12)
        if sharing_weight:
            self.digup.weight = self.embedding.weight
        self.layers = layers
        self.vocab_size, self.hidden_dim, self.learning_rate = vocab_size, hidden_dim, learning_rate
        self.has_pre_layernorm, self.has_post_layernorm = has_pre_layernorm, has_post_layernorm
        self.to_non_negative = to_non_negative

        self.init_weights()

    def init_weights(self) -> None:
        torch.nn.init.xavier_normal_(self.embedding.weight.data)
        torch.nn.init.xavier_normal_(self.digup.weight.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_features = self.embedding(input)

        prior = torch.ones_like(input).unsqueeze(-1).repeat((1, 1, self.vocab_size))

        for layer in self.layers:
            hidden_features = layer(hidden_features)
            if self.has_pre_layernorm:
                hidden_features = self.pre_layernorm(hidden_features)
            evidence_candidated = self.digup(hidden_features)
            if self.has_post_layernorm:
                evidence_candidated = self.post_layernorm(evidence_candidated)
            evidence = self.to_non_negative(evidence_candidated)
            posterior = bayesian_iteration(prior, evidence)
            prior = posterior

        return posterior

    def training_step(self, batch, batch_idx):
        input, target = batch
        posterior = self.forward(input)
        posterior = posterior.permute((0, 2, 1))
        loss = cross_entropy_loss(posterior, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def bayesian_iteration(prior: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    total_evidence = torch.sum(prior * evidence, dim=-1, keepdim=True)
    posterior = (prior * evidence) / total_evidence
    return posterior


def cross_entropy_loss(posterior: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_posterior = torch.log(posterior)
    return F.nll_loss(log_posterior, target)


