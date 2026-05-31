"""MobilePortrait MLX inference pipeline glue: KPDetector(NK) -> DenseMotion -> Inpainting.

Composes the parity-validated MLX modules into one source+driving -> prediction forward,
mirroring the inference path of src/modules/model.py::GeneratorFullModel.forward
(lines 134-163): kp_source/kp_driving = kp_extractor(.), dense_motion(source, kp_d, kp_s),
inpainting(source, dense_motion). bg_param=None, dropout off (inference).

All NHWC. build_from_torch() transfers a torch (kp, dm, inpaint) triple key-for-key so the
whole pipeline can be parity-checked end-to-end against the torch composition.
"""

import mlx.nn as nn

from keypoint_detector_mlx import KPDetector, load_kpdetector_from_torch
from dense_motion_mlx import DenseMotionNetwork, load_dm_from_torch
from inpainting_network_mlx import InpaintingNetwork, load_inpaint_from_torch


class MobilePortraitPipeline(nn.Module):
    def __init__(self, kp, dense_motion, inpainting):
        super().__init__()
        self.kp_extractor = kp
        self.dense_motion_network = dense_motion
        self.inpainting_network = inpainting

    def __call__(self, source, driving):
        # source, driving: NHWC
        kp_source = self.kp_extractor(source)
        kp_driving = self.kp_extractor(driving)
        return self.forward_with_kp(source, kp_source, kp_driving)

    def forward_with_kp(self, source, kp_source, kp_driving):
        """Forward from precomputed keypoint dicts (skips the kp extractor).

        Lets training inject mixed keypoints whose frozen-FK part was computed eagerly
        outside the differentiable graph (the FK detector is a non-traceable onnxruntime
        roundtrip). kp_source/kp_driving must each carry 'fg_kp'. Everything else matches
        __call__."""
        dense_motion = self.dense_motion_network(
            source, kp_driving=kp_driving, kp_source=kp_source, bg_param=None
        )
        generated = self.inpainting_network(source, dense_motion)
        generated["kp_source"] = kp_source
        generated["kp_driving"] = kp_driving
        return generated


def build_from_torch(kp_cfg, dm_cfg, inp_cfg, torch_kp_sd, torch_dm_sd, torch_inp_sd):
    """Construct the MLX pipeline and load identical torch weights into each sub-module.
    Returns (pipeline, drift) where drift maps name -> (missing, unexpected)."""
    kp = KPDetector(**kp_cfg)
    dm = DenseMotionNetwork(**dm_cfg)
    inp = InpaintingNetwork(**inp_cfg)
    load_kpdetector_from_torch(kp, torch_kp_sd)
    dm_miss, dm_unexp = load_dm_from_torch(dm, torch_dm_sd)
    inp_miss, inp_unexp = load_inpaint_from_torch(inp, torch_inp_sd)
    pipe = MobilePortraitPipeline(kp, dm, inp)
    pipe.eval()
    drift = {"dm": (dm_miss, dm_unexp), "inp": (inp_miss, inp_unexp)}
    return pipe, drift
