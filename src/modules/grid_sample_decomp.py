"""Core AI-convertible decomposition of F.grid_sample.

Replaces aten.grid_sampler_2d (which coreai_torch's converter rejects) with
floor/clamp/gather/arithmetic that lower to Core AI ops. Validated 2026-06-09:
  - parity vs F.grid_sample(bilinear, align_corners=True, padding='zeros'):
    worst maxdiff 4.77e-7 across in-bounds / OOB-padding / 64->256 upsample.
  - patched into the real dm+inpaint model: output maxdiff 2.54e-4 (== inpainting
    intrinsic tol), grid_sampler_2d eliminated, the introduced aten.gather is
    ACCEPTED by the Core AI converter. Only remaining blocker = aten.linalg_inv_ex
    (the TPS solve — hoist that out of the graph to fully convert).

Apply by monkeypatching before export/inference:
    import torch.nn.functional as F
    from modules.grid_sample_decomp import grid_sample_bilinear
    F.grid_sample = grid_sample_bilinear   # patches dense_motion:151, inpainting:105

Covers bilinear only (all model sites are bilinear). align_corners True/False and
padding_mode zeros/border supported; model sites use align_corners=True, zeros.
"""

import torch


def grid_sample_bilinear(x, grid, align_corners=True, padding_mode="zeros"):
    # x: (N,C,H,W); grid: (N,Ho,Wo,2) normalized to [-1,1], last dim = (gx, gy)
    N, C, H, W = x.shape
    _, Ho, Wo, _ = grid.shape
    gx, gy = grid[..., 0], grid[..., 1]
    if align_corners:
        ix = (gx + 1) * 0.5 * (W - 1)
        iy = (gy + 1) * 0.5 * (H - 1)
    else:
        ix = ((gx + 1) * W - 1) * 0.5
        iy = ((gy + 1) * H - 1) * 0.5

    x0, y0 = torch.floor(ix), torch.floor(iy)
    x1, y1 = x0 + 1, y0 + 1
    wx1, wy1 = ix - x0, iy - y0
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1

    if padding_mode == "zeros":

        def inb(a, hi):
            return ((a >= 0) & (a <= hi)).to(x.dtype)

        vx0, vx1 = inb(x0, W - 1), inb(x1, W - 1)
        vy0, vy1 = inb(y0, H - 1), inb(y1, H - 1)
    else:  # border: clamp already handles it, no zeroing
        vx0 = vx1 = vy0 = vy1 = torch.ones_like(ix)

    x0c, x1c = x0.clamp(0, W - 1).long(), x1.clamp(0, W - 1).long()
    y0c, y1c = y0.clamp(0, H - 1).long(), y1.clamp(0, H - 1).long()
    xf = x.reshape(N, C, H * W)

    def gather(yc, xc):
        idx = (yc * W + xc).reshape(N, 1, Ho * Wo).expand(N, C, Ho * Wo)
        return torch.gather(xf, 2, idx).reshape(N, C, Ho, Wo)

    def wgt(wy, wx, vy, vx):
        return (wy * wx * vy * vx).unsqueeze(1)  # (N,1,Ho,Wo) broadcast over C

    return (
        gather(y0c, x0c) * wgt(wy0, wx0, vy0, vx0)
        + gather(y0c, x1c) * wgt(wy0, wx1, vy0, vx1)
        + gather(y1c, x0c) * wgt(wy1, wx0, vy1, vx0)
        + gather(y1c, x1c) * wgt(wy1, wx1, vy1, vx1)
    )
