"""MobilePortraitDataset test — verify Δ3/Δ4 fields are attached with correct shapes, that the
sample feeds GeneratorFullModel and fires all 6 losses, and that precompute_multiview produces a
multiview_feats tensor the synthesis forward accepts. All on CPU with stub providers.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import torch

from modules.keypoint_detector import MixedKPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork
from modules.fk_detector import FKDetector
from modules.model import GeneratorFullModel
from modules.mp_dataset import MobilePortraitDataset, precompute_multiview

S = 64  # tiny for CPU speed


class _FakeBase(torch.utils.data.Dataset):
    """Stand-in for TPS FramesDataset: yields source/driving CHW float32 like the real one."""

    def __len__(self):
        return 3

    def __getitem__(self, i):
        rng = np.random.RandomState(i)
        return {
            "source": rng.rand(3, S, S).astype("float32"),
            "driving": rng.rand(3, S, S).astype("float32"),
            "name": f"clip{i}",
        }


DM = dict(
    block_expansion=64,
    max_features=1024,
    num_blocks=5,
    num_tps=10,
    num_channels=3,
    scale_factor=0.25,
    bg=False,
    multi_mask=True,
)
GEN = dict(
    num_channels=3,
    block_expansion=64,
    max_features=512,
    num_down_blocks=3,
    multi_mask=True,
)
TRAIN = dict(
    scales=[1, 0.5, 0.25, 0.125],
    transform_params=dict(sigma_affine=0.05, sigma_tps=0.005, points_tps=5),
    loss_weights=dict(
        perceptual=[10, 10, 10, 10, 10],
        equivariance_value=10,
        warp_loss=10,
        bg=0,
        kp_distance=10,
        landmark_mask=10,
        fg_mask=10,
    ),
    dropout_epoch=35,
    dropout_maxp=0.3,
    dropout_inc_epoch=10,
    dropout_startp=0.1,
    bg_start=10,
)


def main():
    fk = FKDetector(backend="stub")
    ds = MobilePortraitDataset(_FakeBase(), fk_detector=fk)
    s = ds[0]
    assert s["fg_mask"].shape == (1, S, S), s["fg_mask"].shape
    assert s["lmk_mask"].shape == (1, S, S), s["lmk_mask"].shape
    assert s["pseudo_bg"].shape == (3, S, S), s["pseudo_bg"].shape
    assert s["source_fg_mask"].shape == (1, S, S), s["source_fg_mask"].shape
    print(
        f"dataset fields OK: fg{tuple(s['fg_mask'].shape)} lmk{tuple(s['lmk_mask'].shape)} "
        f"bg{tuple(s['pseudo_bg'].shape)}"
    )

    # collate a batch and run the full model — all 6 losses must fire
    batch = torch.utils.data.dataloader.default_collate([ds[0], ds[1]])
    torch.manual_seed(0)
    model = GeneratorFullModel(
        MixedKPDetector(num_tps=10, fk_backend="stub"),
        None,
        DenseMotionNetwork(**DM),
        InpaintingNetwork(**GEN),
        TRAIN,
    ).train()
    losses, _ = model(batch, epoch=40)
    for k in (
        "perceptual",
        "equivariance_value",
        "warp_loss",
        "kp",
        "landmark",
        "mask",
    ):
        assert k in losses, f"missing loss {k}"
    print(
        "dataset -> GeneratorFullModel: all 6 losses fire:",
        {k: round(float(v), 3) for k, v in losses.items()},
    )

    # precompute_multiview shape
    inp = InpaintingNetwork(**GEN)
    drv_frames = [np.random.rand(3, S, S).astype("float32") for _ in range(6)]
    mv = precompute_multiview(
        inp,
        s["source"],
        drv_frames,
        num_views=4,
        pseudo_bg=s["pseudo_bg"],
        fg_mask=s["source_fg_mask"],
    )
    low = inp.low_ch
    h_low = S // (2 ** GEN["num_down_blocks"])
    assert mv.shape == (1, 4, low, h_low, h_low), mv.shape
    print(f"precompute_multiview OK: {tuple(mv.shape)}")
    print("MobilePortraitDataset — ALL PASS")
    open("/tmp/mp_dataset_result.txt", "w").write("ALL PASS\n")


if __name__ == "__main__":
    main()
