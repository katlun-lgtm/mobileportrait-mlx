"""Parity gate for the Core AI-convertible grid_sample decomposition vs F.grid_sample.
Run: python src/tests/test_grid_sample_decomp_parity.py  (exits 0 on ALL_PASS)."""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
from grid_sample_decomp import grid_sample_bilinear


def _case(name, N, C, H, W, Ho, Wo, scale):
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W)
    grid = (torch.rand(N, Ho, Wo, 2) * 2 - 1) * scale  # scale>1 forces OOB -> zeros pad
    ref = F.grid_sample(
        x, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    got = grid_sample_bilinear(x, grid, align_corners=True, padding_mode="zeros")
    d = (ref - got).abs().max().item()
    print(f"{name:<22} maxdiff={d:.3e}  {'PASS' if d < 1e-4 else 'FAIL'}")
    return d


def main():
    ds = [
        _case("inbounds 3ch 64", 2, 3, 64, 64, 64, 64, 0.9),
        _case("OOB-padding 3ch 64", 2, 3, 64, 64, 64, 64, 1.6),
        _case("inbounds 8ch 32", 2, 8, 32, 32, 32, 32, 0.95),
        _case("OOB 11ch 64 (dm-like)", 2, 11, 64, 64, 64, 64, 1.4),
        _case("upsample-warp 64->256", 1, 64, 64, 64, 256, 256, 1.1),
    ]
    ok = max(ds) < 1e-4
    print(f"\nRESULT {'ALL_PASS' if ok else 'FAIL'}  worst={max(ds):.3e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
