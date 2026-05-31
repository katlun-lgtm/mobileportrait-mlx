"""MixedKP MLP — MobilePortrait Δ1: fuse 106 FK + 50 NK -> 50 mixed keypoints.

concat(FK 106*2=212, NK 50*2=100) = 312 -> MLP -> 100 -> reshape (50,2) = delta.
RESIDUAL form: mixed = nk + delta. With the final layer zero-initialised the delta is 0
at init, so mixed == nk -- i.e. MixedKP starts as an identity passthrough of the (warm-
started) neural keypoints. That preserves the warm-started DenseMotion/Inpainting instead
of scrambling it with a random fusion head, which is what made the from-scratch tanh(mlp)
variant start ~2x worse. The mixed keypoints replace the neural keypoints fed to TPS/DMN.
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
        residual: bool = True,
    ):
        super().__init__()
        layers, d = [], num_fk * 2 + num_nk * 2
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, num_mixed * 2)]
        self.mlp = nn.Sequential(*layers)
        self.num_mixed = num_mixed
        self.residual = residual
        if residual:
            # delta=0 at init -> mixed == nk (identity passthrough; keeps warm-start)
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, fk: torch.Tensor, nk: torch.Tensor) -> torch.Tensor:
        """fk (bs,106,2), nk (bs,num_mixed,2) -> (bs,num_mixed,2)."""
        bs = fk.shape[0]
        x = torch.cat([fk.reshape(bs, -1), nk.reshape(bs, -1)], dim=1)
        delta = self.mlp(x).view(bs, self.num_mixed, 2)
        if self.residual:
            return nk + delta
        return torch.tanh(delta)
