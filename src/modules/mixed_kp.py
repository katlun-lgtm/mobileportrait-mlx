"""MixedKP MLP — MobilePortrait Δ1: fuse 106 FK + 50 NK -> 50 mixed keypoints.

concat(FK 106*2=212, NK 50*2=100) = 312 -> MLP -> 100 -> reshape (50,2), tanh to [-1,1].
The mixed keypoints replace the neural keypoints fed to TPS/DMN downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MixedKP(nn.Module):
    def __init__(
        self,
        num_fk: int = 106,
        num_nk: int = 50,
        num_mixed: int = 50,
        hidden=(256, 256),
    ):
        super().__init__()
        din = num_fk * 2 + num_nk * 2
        layers, d = [], din
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, num_mixed * 2)]
        self.mlp = nn.Sequential(*layers)
        self.num_mixed = num_mixed

    def forward(self, fk: torch.Tensor, nk: torch.Tensor) -> torch.Tensor:
        """fk (bs,106,2), nk (bs,50,2) -> (bs,num_mixed,2) in [-1,1]."""
        bs = fk.shape[0]
        x = torch.cat([fk.reshape(bs, -1), nk.reshape(bs, -1)], dim=1)
        return torch.tanh(self.mlp(x)).view(bs, self.num_mixed, 2)
