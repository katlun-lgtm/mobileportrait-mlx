"""MLX/NHWC port of src/modules/inpainting_network.py::InpaintingNetwork (with Δ4).

Reuses blocks_mlx (SameBlock2d, DownBlock2d, UpBlock2d, ResBlock2d) + grid_sample_mlx.
Channels-last throughout. Named to mirror the torch module so a torch state_dict
transfers key-for-key (down_blocks/up_blocks/resblock are reversed at build time exactly
like torch, so list indices align). load_inpaint_from_torch() reports key drift.

deform_input bilinearly resizes the deformation grid to the feature resolution (matching
torch F.interpolate(mode='bilinear', align_corners=True) on the (B,2,h,w) grid) then runs
grid_sample (align_corners=True, zeros) — same as the reference.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from blocks_mlx import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from util_mlx import make_coordinate_grid
from grid_sample_mlx import make_grid_sample_2d

_gs = make_grid_sample_2d(align_corners=True, padding_mode="zeros")


def _resize_grid_bilinear(grid, h, w):
    """grid (B,h0,w0,2) -> (B,h,w,2), bilinear align_corners=True via grid_sample.
    Treat the 2-channel grid as an image and sample it on the target identity lattice."""
    b, h0, w0, _ = grid.shape
    if h0 == h and w0 == w:
        return grid
    target = make_coordinate_grid((h, w), grid.dtype)  # (h,w,2) in [-1,1], (x,y)
    target = mx.broadcast_to(target[None], (b, h, w, 2))
    # grid is a (B,h0,w0,C=2) "image"; sample it -> (B,h,w,2)
    return _gs(grid, target)


class InpaintingNetwork(nn.Module):
    def __init__(
        self,
        num_channels,
        block_expansion,
        max_features,
        num_down_blocks,
        multi_mask=True,
        num_multiview=4,
        use_pseudo_bg=True,
        **kwargs,
    ):
        super().__init__()
        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.num_multiview = num_multiview
        self.use_pseudo_bg = use_pseudo_bg
        first_in = num_channels + 4 if use_pseudo_bg else num_channels
        self.first = SameBlock2d(
            first_in, block_expansion, kernel_size=(7, 7), padding=(3, 3)
        )

        down_blocks, up_blocks, resblock = [], [], []
        for i in range(num_down_blocks):
            in_f = min(max_features, block_expansion * (2**i))
            out_f = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(
                DownBlock2d(in_f, out_f, kernel_size=(3, 3), padding=(1, 1))
            )
            dec_in = out_f * 2
            if i == num_down_blocks - 1:
                dec_in = out_f
            up_blocks.append(
                UpBlock2d(dec_in, in_f, kernel_size=(3, 3), padding=(1, 1))
            )
            resblock.append(ResBlock2d(dec_in, kernel_size=3, padding=1))
            resblock.append(ResBlock2d(dec_in, kernel_size=3, padding=1))
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks[::-1]
        self.resblock = resblock[::-1]

        self.final = nn.Conv2d(
            block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3)
        )
        self.num_channels = num_channels
        self.low_ch = min(max_features, block_expansion * (2**num_down_blocks))
        self.mv_merge = nn.Conv2d(
            self.low_ch * 2, self.low_ch, kernel_size=(3, 3), padding=(1, 1)
        )

    def _augment(self, image, pseudo_bg=None, fg_mask=None):
        if not self.use_pseudo_bg:
            return image
        if pseudo_bg is None:
            pseudo_bg = mx.zeros_like(image)
        if fg_mask is None:
            fg_mask = mx.zeros(
                (image.shape[0], image.shape[1], image.shape[2], 1), dtype=image.dtype
            )
        return mx.concatenate([image, pseudo_bg, fg_mask], axis=-1)

    def deform_input(self, inp, deformation):
        # inp NHWC, deformation (B,h0,w0,2)
        h, w = inp.shape[1], inp.shape[2]
        deformation = _resize_grid_bilinear(deformation, h, w)
        return _gs(inp, deformation)

    def occlude_input(self, inp, occ):
        # occ is (B,1,h,w) torch-style -> NHWC (B,h,w,1)
        return inp * occ

    def __call__(
        self,
        source_image,
        dense_motion,
        multiview_feats=None,
        pseudo_bg=None,
        fg_mask=None,
    ):
        out = self.first(self._augment(source_image, pseudo_bg, fg_mask))
        encoder_map = [out]
        for d in self.down_blocks:
            out = d(out)
            encoder_map.append(out)

        if multiview_feats is not None and self.num_multiview > 0:
            mv = mx.mean(multiview_feats, axis=1)
            out = self.mv_merge(mx.concatenate([out, mv], axis=-1))
            encoder_map[-1] = out

        out_dict = {}
        out_dict["contribution_maps"] = dense_motion["contribution_maps"]
        out_dict["deformed_source"] = dense_motion["deformed_source"]
        occlusion_map = dense_motion["occlusion_map"]  # list of (B,h,w,1) NHWC
        out_dict["occlusion_map"] = occlusion_map
        deformation = dense_motion["deformation"]

        out = self.deform_input(out, deformation)
        out = self.occlude_input(out, occlusion_map[0])

        for i in range(self.num_down_blocks):
            out = self.resblock[2 * i](out)
            out = self.resblock[2 * i + 1](out)
            out = self.up_blocks[i](out)

            encode_i = encoder_map[-(i + 2)]
            encode_i = self.deform_input(encode_i, deformation)
            occ_ind = (i + 1) if self.multi_mask else 0
            encode_i = self.occlude_input(encode_i, occlusion_map[occ_ind])

            if i == self.num_down_blocks - 1:
                break
            out = mx.concatenate([out, encode_i], axis=-1)

        deformed_source = self.deform_input(source_image, deformation)
        out_dict["deformed"] = deformed_source

        occlusion_last = occlusion_map[-1]
        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)
        out = mx.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        out_dict["prediction"] = out
        return out_dict


def load_inpaint_from_torch(mlx_mod, torch_sd):
    mlx_keys = set(k for k, _ in tree_flatten(mlx_mod.parameters()))
    flat, torch_keys = [], set()
    for k, v in torch_sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        torch_keys.add(k)
        arr = mx.array(v.detach().cpu().numpy())
        if arr.ndim == 4:
            arr = mx.transpose(arr, (0, 2, 3, 1))
        flat.append((k, arr))
    missing = sorted(mlx_keys - torch_keys)
    unexpected = sorted(torch_keys - mlx_keys)
    mlx_mod.unfreeze(recurse=True)
    mlx_mod.update(tree_unflatten(flat))
    mlx_mod.eval()
    return missing, unexpected
