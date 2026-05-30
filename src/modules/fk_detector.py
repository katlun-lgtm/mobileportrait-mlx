"""Frozen 106-point facial-keypoint (FK) detector — MobilePortrait Δ1.

Real backend = insightface 2d106 (or mediapipe). Until that's wired on the GPU/training
box, a deterministic 'stub' returns (bs,106,2) in [-1,1] so the rest of the pipeline
(MixedKP MLP, DMN) can be built + shape-tested on CPU. FK is FROZEN — never trained;
runs as a preprocessing step (no grad).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FKDetector(nn.Module):
    def __init__(self, backend: str = "stub", num_fk: int = 106):
        super().__init__()
        self.num_fk = num_fk
        self.backend = backend
        self._app = None
        if backend == "insightface":
            self._init_insightface()

    def _init_insightface(self):
        from insightface.app import FaceAnalysis  # lazy: only on training box

        self._app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"]
        )
        # ctx_id=-1 → CPU providers (Mac has no CUDA onnxruntime); FK is cheap one-shot preproc.
        self._app.prepare(ctx_id=-1, det_size=(256, 256))

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image (bs,3,H,W) in [0,1] -> (bs,num_fk,2) normalized to [-1,1]."""
        bs = image.shape[0]
        if self.backend == "stub":
            g = torch.linspace(-0.8, 0.8, self.num_fk, device=image.device)
            pts = torch.stack(
                [g, g.flip(0)], dim=-1
            )  # (num_fk,2) placeholder, NOT real
            return pts.unsqueeze(0).repeat(bs, 1, 1)
        # insightface path
        import cv2
        import numpy as np

        out = torch.zeros(bs, self.num_fk, 2, device=image.device)
        imgs = (
            (image.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # bs,H,W,3 RGB
        H, W = imgs.shape[1:3]
        for i, im in enumerate(imgs):
            faces = self._app.get(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            if faces:
                # largest face by bbox area
                f = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                lmk = np.asarray(f.landmark_2d_106, dtype="float32")  # (106,2) pixels
                norm = lmk / np.array([W, H], dtype="float32") * 2 - 1
                out[i] = torch.from_numpy(norm).to(out.device)
        return out
