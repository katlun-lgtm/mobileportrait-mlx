"""MobilePortrait dataset wrapper — adds the Δ3/Δ4 training targets on top of TPS FramesDataset.

TPS `FramesDataset` yields `{'source','driving','name'}` (CHW float32 in [0,1]). MobilePortrait
also needs, per training pair:

  Δ3  driving 'fg_mask'   (1,H,W)  — foreground segmentation of the driving frame   (L_mask)
  Δ3  driving 'lmk_mask'  (1,H,W)  — rasterised facial-landmark heatmap of driving   (L_landmark)
  Δ4b source  'pseudo_bg' (3,H,W)  — LaMa-inpainted background of the source          (synthesis)
  Δ4b source  'source_fg_mask' (1,H,W) — foreground mask of the source (BG composite)

Providers are pluggable callables; each defaults to a cheap CPU stub so the pipeline runs on the
dev box with nothing installed. On the training box, pass real providers (MODNet/rembg segmenter,
LaMa inpainter, the FK detector's landmarks → landmark_mask_from_points). Pseudo-multiview feats
are NOT produced here — they need the model encoder, so they're precomputed once per source in a
separate offline pass (see precompute_multiview docstring).

Whatever a provider returns is attached to the sample; missing fields are simply absent, and the
model's Δ3/Δ4 code already guards on presence, so partial wiring degrades gracefully.
"""

import numpy as np
import torch

from modules.model import landmark_mask_from_points


# ----------------------------------------------------------------------------- stub providers
def stub_fg_mask(image_chw):
    """Cheap fg mask placeholder: center ellipse. Replace with MODNet/rembg on the train box.
    image_chw: (3,H,W) float32 [0,1] -> (1,H,W) float32 {0,1}."""
    _, h, w = image_chw.shape
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    cy, cx = h / 2, w / 2
    ell = ((yy - cy) / (0.45 * h)) ** 2 + ((xx - cx) / (0.35 * w)) ** 2
    return (ell <= 1.0).astype("float32")[None]


def stub_pseudo_bg(image_chw, fg_mask_1hw):
    """Cheap pseudo-BG placeholder: blur-fill the fg region with the border mean.
    Replace with LaMa on the train box. Returns (3,H,W) float32 [0,1]."""
    img = image_chw.copy()
    bg = fg_mask_1hw[0] < 0.5
    if bg.any():
        fill = img[:, bg].mean(axis=1, keepdims=True)  # (3,1) mean of background pixels
    else:
        fill = img.reshape(3, -1).mean(axis=1, keepdims=True)
    out = img.copy()
    fgm = fg_mask_1hw[0] >= 0.5
    out[:, fgm] = fill
    return out.astype("float32")


# ----------------------------------------------------------------------------- main wrapper
class MobilePortraitDataset(torch.utils.data.Dataset):
    """Wrap a TPS FramesDataset (train mode) and attach Δ3/Δ4 targets.

    Args:
      base                : a TPS FramesDataset (is_train=True) yielding source/driving CHW.
      fk_detector         : frozen FK detector image(1,3,H,W)->(1,N,2) for the driving landmark
                            mask. If None, lmk_mask is skipped (L_landmark won't fire).
      seg_provider(chw)   : (3,H,W)->(1,H,W) fg mask. Default = stub ellipse.
      bg_provider(chw,m)  : (3,H,W),(1,H,W)->(3,H,W) pseudo-BG. Default = stub border-fill.
      emit_pseudo_bg      : attach source pseudo_bg + source_fg_mask (Δ4b). Default True.
      emit_driving_masks  : attach driving fg_mask + lmk_mask (Δ3). Default True.
    """

    def __init__(
        self,
        base,
        fk_detector=None,
        seg_provider=None,
        bg_provider=None,
        emit_pseudo_bg=True,
        emit_driving_masks=True,
    ):
        self.base = base
        self.fk_detector = fk_detector
        self.seg = seg_provider or stub_fg_mask
        self.bg = bg_provider or stub_pseudo_bg
        self.emit_pseudo_bg = emit_pseudo_bg
        self.emit_driving_masks = emit_driving_masks

    def __len__(self):
        return len(self.base)

    def _landmarks(self, image_chw):
        """Run the FK detector on one CHW image -> (N,2) normalised, or None."""
        if self.fk_detector is None:
            return None
        t = torch.from_numpy(image_chw[None])  # (1,3,H,W)
        with torch.no_grad():
            fk = self.fk_detector(t)  # (1,N,2)
        return fk[0]

    def __getitem__(self, idx):
        out = self.base[idx]
        if "driving" not in out:  # not a train-mode sample; pass through
            return out
        source = out["source"]  # (3,H,W) float32
        driving = out["driving"]
        _, h, w = driving.shape

        if self.emit_driving_masks:
            out["fg_mask"] = self.seg(driving)  # (1,H,W)
            lmk = self._landmarks(driving)
            if lmk is not None:
                out["lmk_mask"] = (
                    landmark_mask_from_points(lmk[None], size=h)[0]
                    .numpy()
                    .astype("float32")
                )  # (1,H,W)

        if self.emit_pseudo_bg:
            src_fg = self.seg(source)  # (1,H,W)
            out["source_fg_mask"] = src_fg
            out["pseudo_bg"] = self.bg(source, src_fg)  # (3,H,W)

        return out


@torch.no_grad()
def precompute_multiview(
    inpainting_network,
    source_chw,
    driving_frames_chw,
    num_views=4,
    pseudo_bg=None,
    fg_mask=None,
    device="cpu",
):
    """Δ4a: precompute the pseudo-multiview features for ONE source (offline, once per source).

    Picks `num_views` uniformly-spaced driving frames, encodes each warped-source view to the
    lowest downblock via `inpainting_network.encode_lowest`, and stacks them as the
    `multiview_feats` tensor (1,T,low_ch,h_low,w_low) the synthesis forward expects.

    NOTE this is a simplified stand-in: the paper warps the source toward each driving frame via
    the motion module before encoding. Here we encode the driving frames directly as the
    appearance views — adequate to wire shapes and the offline-cache path on CPU; swap in the
    motion-warped variant on the train box once the full pipeline runs end-to-end.
    """
    src = torch.as_tensor(source_chw, device=device)[None]  # (1,3,H,W)
    n = len(driving_frames_chw)
    idxs = np.linspace(0, n - 1, num=min(num_views, n)).round().astype(int)
    bg = None if pseudo_bg is None else torch.as_tensor(pseudo_bg, device=device)[None]
    fm = None if fg_mask is None else torch.as_tensor(fg_mask, device=device)[None]
    feats = []
    for i in idxs:
        view = torch.as_tensor(driving_frames_chw[i], device=device)[None]
        feats.append(inpainting_network.encode_lowest(view, bg, fm))  # (1,low_ch,h,w)
    return torch.stack(feats, dim=1)  # (1,T,low_ch,h_low,w_low)
