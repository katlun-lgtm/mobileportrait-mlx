"""MLX/NHWC port of src/modules/dense_motion.py::DenseMotionNetwork (with Δ2/Δ3).

Reuses blocks_mlx (Hourglass, AntiAliasInterpolation2d, UpBlock2d), util_mlx
(TPS, kp2gaussian, make_coordinate_grid, to/from_homogeneous), and grid_sample_mlx.

Channels-last throughout. Named to mirror the torch module so a torch state_dict
transfers key-for-key. load_dm_from_torch() reports any key-set drift.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from blocks_mlx import Hourglass, AntiAliasInterpolation2d, UpBlock2d
from util_mlx import (
    TPS,
    kp2gaussian,
    make_coordinate_grid,
    to_homogeneous,
    from_homogeneous,
)
from grid_sample_mlx import make_grid_sample_2d

_grid_sample = make_grid_sample_2d(align_corners=True, padding_mode="zeros")


class DenseMotionNetwork(nn.Module):
    def __init__(
        self,
        block_expansion,
        num_blocks,
        max_features,
        num_tps,
        num_channels,
        scale_factor=0.25,
        bg=False,
        multi_mask=True,
        kp_variance=0.01,
    ):
        super().__init__()
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1),
            max_features=max_features,
            num_blocks=num_blocks,
        )
        out = self.hourglass.out_channels
        self.maps = nn.Conv2d(out[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))

        if multi_mask:
            self.up_nums = int(math.log(1 / scale_factor, 2))
            self.occlusion_num = 4
            channel = [out[-1] // (2**i) for i in range(self.up_nums)]
            up = [
                UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1)
                for i in range(self.up_nums)
            ]
            self.up = up
            channel = [
                out[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]
            ]
            for i in range(self.up_nums):
                channel.append(out[-1] // (2 ** (i + 1)))
            self.occlusion = [
                nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3))
                for i in range(self.occlusion_num)
            ]
        else:
            self.occlusion = [nn.Conv2d(out[-1], 1, kernel_size=(7, 7), padding=(3, 3))]

        self.num_tps = num_tps
        self.bg = bg
        self.kp_variance = kp_variance

        feat_ch = out[-1]
        self.residual_flow = nn.Conv2d(feat_ch, 2, kernel_size=(7, 7), padding=(3, 3))
        self.fg_mask_head = nn.Conv2d(feat_ch, 1, kernel_size=(7, 7), padding=(3, 3))
        self.lmk_mask_head = nn.Conv2d(feat_ch, 1, kernel_size=(7, 7), padding=(3, 3))

    # --- sub-steps (all NHWC) ---
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        h, w = source_image.shape[1], source_image.shape[2]
        g_d = kp2gaussian(kp_driving["fg_kp"], (h, w), self.kp_variance)  # (bs,K,h,w)
        g_s = kp2gaussian(kp_source["fg_kp"], (h, w), self.kp_variance)
        heatmap = mx.transpose(g_d - g_s, (0, 2, 3, 1))  # (bs,h,w,K)
        zeros = mx.zeros((heatmap.shape[0], h, w, 1), dtype=heatmap.dtype)
        return mx.concatenate([zeros, heatmap], axis=-1)  # (bs,h,w,K+1)

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        bs, h, w = source_image.shape[0], source_image.shape[1], source_image.shape[2]
        kp_1 = kp_driving["fg_kp"].reshape(bs, -1, 5, 2)
        kp_2 = kp_source["fg_kp"].reshape(bs, -1, 5, 2)
        trans = TPS(mode="kp", bs=bs, kp_1=kp_1, kp_2=kp_2)
        gs = kp_1.shape[1]
        grid = make_coordinate_grid((h, w), kp_1.dtype).reshape(1, h * w, 2)
        driving_to_source = trans.warp_coordinates(grid).reshape(bs, gs, h, w, 2)

        identity = make_coordinate_grid((h, w), kp_1.dtype).reshape(1, 1, h, w, 2)
        identity = mx.broadcast_to(identity, (bs, 1, h, w, 2))
        if bg_param is not None:
            identity = to_homogeneous(identity)
            identity = (bg_param.reshape(bs, 1, 1, 1, 3, 3) @ identity[..., None])[
                ..., 0
            ]
            identity = from_homogeneous(identity)
        return mx.concatenate([identity, driving_to_source], axis=1)  # (bs,K+1,h,w,2)

    def create_deformed_source_image(self, source_image, transformations):
        bs, h, w, c = source_image.shape
        k1 = self.num_tps + 1
        src = mx.broadcast_to(source_image[:, None], (bs, k1, h, w, c)).reshape(
            bs * k1, h, w, c
        )
        grid = transformations.reshape(bs * k1, h, w, 2)
        deformed = _grid_sample(src, grid)  # (bs*k1,h,w,c)
        return deformed.reshape(bs, k1, h, w, c)

    def __call__(self, source_image, kp_driving, kp_source, bg_param=None):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
        bs, h, w, c = source_image.shape
        out_dict = {}

        heatmap = self.create_heatmap_representations(
            source_image, kp_driving, kp_source
        )
        transformations = self.create_transformations(
            source_image, kp_driving, kp_source, bg_param
        )
        deformed_source = self.create_deformed_source_image(
            source_image, transformations
        )
        out_dict["deformed_source"] = deformed_source

        # torch view(bs,-1,h,w) from (bs,K+1,C,h,w): channel = k*C + c. NHWC equivalent:
        k1 = self.num_tps + 1
        deformed_cat = mx.transpose(deformed_source, (0, 2, 3, 1, 4)).reshape(
            bs, h, w, k1 * c
        )
        net_in = mx.concatenate([heatmap, deformed_cat], axis=-1)  # (bs,h,w,51+(K+1)C)

        prediction = self.hourglass(net_in, mode=1)  # list of NHWC feats
        feat = prediction[-1]

        contribution_maps = self.maps(feat)  # (bs,h,w,K+1)
        contribution_maps = mx.softmax(contribution_maps, axis=-1)
        out_dict["contribution_maps"] = contribution_maps

        # deformation = sum_k cmap_k * transformations_k
        cmap = contribution_maps[..., None]  # (bs,h,w,K+1,1)
        trans_t = mx.transpose(transformations, (0, 2, 3, 1, 4))  # (bs,h,w,K+1,2)
        deformation = mx.sum(trans_t * cmap, axis=3)  # (bs,h,w,2)
        deformation = deformation + self.residual_flow(feat)  # Δ2
        out_dict["deformation"] = deformation

        if self.training:
            out_dict["fg_mask_pred"] = mx.sigmoid(self.fg_mask_head(feat))
            out_dict["lmk_mask_pred"] = mx.sigmoid(self.lmk_mask_head(feat))

        occlusion_map = []
        if self.multi_mask:
            for i in range(self.occlusion_num - self.up_nums):
                occlusion_map.append(
                    mx.sigmoid(
                        self.occlusion[i](
                            prediction[self.up_nums - self.occlusion_num + i]
                        )
                    )
                )
            pred = prediction[-1]
            for i in range(self.up_nums):
                pred = self.up[i](pred)
                occlusion_map.append(
                    mx.sigmoid(
                        self.occlusion[i + self.occlusion_num - self.up_nums](pred)
                    )
                )
        else:
            occlusion_map.append(mx.sigmoid(self.occlusion[0](prediction[-1])))
        out_dict["occlusion_map"] = occlusion_map
        return out_dict


def load_dm_from_torch(mlx_dm, torch_sd):
    """Transfer torch DenseMotionNetwork state_dict into the MLX module.
    Returns (missing_in_torch, unexpected_in_torch) key sets for drift checking.
    """
    mlx_keys = set(k for k, _ in tree_flatten(mlx_dm.parameters()))
    flat, torch_keys = [], set()
    for k, v in torch_sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        torch_keys.add(k)
        arr = mx.array(v.detach().cpu().numpy())
        if arr.ndim == 4:
            arr = mx.transpose(arr, (0, 2, 3, 1))
        flat.append((k, arr))
    missing = mlx_keys - torch_keys  # in MLX but not provided by torch
    unexpected = torch_keys - mlx_keys  # provided by torch but no MLX slot
    mlx_dm.unfreeze(recurse=True)
    mlx_dm.update(tree_unflatten(flat))
    mlx_dm.eval()
    return sorted(missing), sorted(unexpected)
