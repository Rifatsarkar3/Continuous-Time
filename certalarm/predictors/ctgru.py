"""CT-GRU: a grid ODE-RNN over the minute grid.

Between minutes the latent state drifts by one Euler step of a learned vector field; at a minute
carrying a measurement a GRU cell updates it, and at an unobserved minute it only drifts. The
head emits a mean in [0, 1] and a strictly positive scale per minute.

This is the base variant. Extended variants were tested earlier and rejected, so they are not
carried over.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTGRU(nn.Module):
    def __init__(self, n_feat, H=32):
        super().__init__()
        self.H = H
        self.cell = nn.GRUCell(n_feat, H)
        self.f = nn.Sequential(nn.Linear(H + 1, H), nn.Tanh(), nn.Linear(H, H))
        self.head = nn.Linear(H + 1, 2)
        self.dt = 0.1

    def forward(self, X, M):
        """X: (B, L, n_feat) normalized features. M: (B, L) 1 where a measurement was acquired."""
        B, L, _ = X.shape
        h = torch.zeros(B, self.H, device=X.device, dtype=X.dtype)
        mus, sigmas = [], []
        for k in range(L):
            tf = torch.full((B, 1), math.log1p(k / 60.0), device=X.device, dtype=X.dtype)
            if k > 0:
                h = h + self.dt * self.f(torch.cat([h, tf], 1))
            m = M[:, k:k + 1]
            h = m * self.cell(X[:, k], h) + (1 - m) * h
            o = self.head(torch.cat([h, tf], 1))
            mus.append(torch.sigmoid(o[:, :1]))
            sigmas.append(F.softplus(o[:, 1:]) + 1e-3)
        return torch.cat(mus, 1), torch.cat(sigmas, 1)
