from torch import nn
import torch
from torchvision import models

from .fk_detector import FKDetector
from .mixed_kp import MixedKP


class KPDetector(nn.Module):
    """
    Predict K*5 neural keypoints (NK). Unchanged from TPS.
    """

    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps * 5 * 2)

    def forward(self, image):

        fg_kp = self.fg_encoder(image)
        (
            bs,
            _,
        ) = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        out = {"fg_kp": fg_kp.view(bs, self.num_tps * 5, -1)}

        return out


class MixedKPDetector(nn.Module):
    """MobilePortrait Δ1 — drop-in replacement for KPDetector.

    NK (learned KPDetector) + frozen 106-pt FK + MixedKP MLP -> 50 mixed keypoints,
    exposed as 'fg_kp' so DenseMotionNetwork / TPS transforms consume them unchanged.
    """

    def __init__(self, num_tps, fk_backend="stub", num_fk=106, **kwargs):
        super(MixedKPDetector, self).__init__()
        self.num_tps = num_tps
        self.nk = KPDetector(num_tps, **kwargs)
        self.fk = FKDetector(backend=fk_backend, num_fk=num_fk)
        self.mixed = MixedKP(num_fk=num_fk, num_nk=num_tps * 5, num_mixed=num_tps * 5)

    def forward(self, image, fk_kp=None):
        nk_kp = self.nk(image)["fg_kp"]  # (bs, num_tps*5, 2)  learned
        if fk_kp is None:
            fk_kp = self.fk(image)  # (bs, num_fk, 2)  frozen; skipped when pre-cached
        else:
            fk_kp = fk_kp.to(image.device)  # cache tensors arrive as CPU
        mixed_kp = self.mixed(fk_kp, nk_kp)  # (bs, num_tps*5, 2)
        return {"fg_kp": mixed_kp, "nk_kp": nk_kp, "fk_kp": fk_kp}
