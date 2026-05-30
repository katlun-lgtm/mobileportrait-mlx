"""Real Δ3/Δ4 providers for the dataset — foreground segmentation + background inpaint.

Each is a callable matching the `MobilePortraitDataset` provider signatures:
  seg_provider(image_chw: np.ndarray (3,H,W) [0,1]) -> (1,H,W) float32 {0,1}
  bg_provider(image_chw, fg_mask_1hw) -> (3,H,W) float32 [0,1]

Backends:
  - rembg (U2Net) for foreground matting -> fg mask.
  - LaMa (simple-lama-inpainting) for background fill behind the removed foreground; falls
    back to OpenCV Telea inpaint if simple-lama isn't installed.

Models load lazily on first call (so importing this module is cheap and dev-safe).
"""

from __future__ import annotations

import numpy as np


class RembgSegProvider:
    """Foreground mask via rembg (U2Net). Returns (1,H,W) float32 in {0,1}."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._session = None

    def _ensure(self):
        if self._session is None:
            from rembg import new_session

            self._session = new_session("u2net")

    def __call__(self, image_chw: np.ndarray) -> np.ndarray:
        self._ensure()
        from rembg import remove

        hwc = (np.clip(image_chw, 0, 1).transpose(1, 2, 0) * 255).astype("uint8")
        matte = remove(hwc, session=self._session, only_mask=True)  # (H,W) uint8
        m = (np.asarray(matte, dtype="float32") / 255.0 >= self.threshold).astype(
            "float32"
        )
        return m[None]  # (1,H,W)


class LamaBgProvider:
    """Pseudo-background: inpaint the foreground region. LaMa if available, else cv2 Telea."""

    def __init__(self):
        self._lama = None
        self._tried = False

    def _ensure(self):
        if self._tried:
            return
        self._tried = True
        try:
            from simple_lama_inpainting import SimpleLama

            self._lama = SimpleLama()
        except Exception:
            self._lama = None

    def __call__(self, image_chw: np.ndarray, fg_mask_1hw: np.ndarray) -> np.ndarray:
        self._ensure()
        hwc = (np.clip(image_chw, 0, 1).transpose(1, 2, 0) * 255).astype("uint8")
        mask = (fg_mask_1hw[0] >= 0.5).astype("uint8") * 255  # (H,W) region to fill
        if self._lama is not None:
            from PIL import Image

            out = self._lama(Image.fromarray(hwc), Image.fromarray(mask))
            arr = np.asarray(out.convert("RGB"), dtype="float32") / 255.0
            return arr.transpose(2, 0, 1)
        import cv2

        bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        return rgb.transpose(2, 0, 1)


def build_providers(kind: str = "real"):
    """Return (seg_provider, bg_provider). kind='real' -> rembg+LaMa; else (None,None)=stubs."""
    if kind == "real":
        return RembgSegProvider(), LamaBgProvider()
    return None, None
