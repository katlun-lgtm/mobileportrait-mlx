"""MLX losses for MobilePortrait training (beyond L1+perceptual).

equivariance_loss: model.py 182-206. Random-TPS-warp the driving frame, re-detect
keypoints on the warped frame, warp those kp back with the SAME TPS, compare to the
original driving kp:  L = |kp_driving - transform.warp_coordinates(kp(warp(driving)))|.mean()

The warped *image* (input to re-detection) uses grid_sample; the reference uses
padding_mode='reflection' but our Metal grid_sample only does zeros/border. We use
'border' — reflection vs border differ only in a 1px boundary strip, negligible for
keypoint detection, and the LOSS itself is a coordinate comparison (warp_coordinates,
already parity-validated in util_mlx.TPS), not an image comparison. So the faithful part
is exact; only the re-detection input's extreme border differs微.

equivariance_warp_grid(): builds the random-TPS sampling grid for the image warp, in the
NHWC (x,y) convention grid_sample_mlx expects.
"""

import mlx.core as mx
from util_mlx import TPS
from grid_sample_mlx import make_grid_sample_2d

_gs_border = make_grid_sample_2d(align_corners=True, padding_mode="border")


def equivariance_loss(
    kp_extractor, driving, kp_driving, transform_params, noise=None, control_params=None
):
    """driving: NHWC. kp_driving: dict with 'fg_kp' (bs, K, 2). Returns scalar mx.array.

    noise/control_params: optional explicit random params (for parity vs torch); if None,
    sampled internally via mx.random.
    """
    bs = driving.shape[0]
    kw = dict(transform_params)
    if noise is not None:
        kw["_noise"] = noise
    if control_params is not None:
        kw["_control_params"] = control_params
    transform = TPS(mode="random", bs=bs, **kw)

    h, w = driving.shape[1], driving.shape[2]
    # transform_frame reads frame.shape[2],[3] as (H,W) (NCHW convention) and returns
    # (bs,h,w,2) in random mode. Pass a (bs,C,h,w)-shaped stand-in so H,W are read right.
    grid = transform.transform_frame(mx.zeros((bs, 3, h, w)))  # (bs,h,w,2), (x,y)
    transformed = _gs_border(driving, grid)
    transformed_kp = kp_extractor(transformed)["fg_kp"]

    warped = transform.warp_coordinates(transformed_kp)  # (bs, K, 2)
    return mx.mean(mx.abs(kp_driving["fg_kp"] - warped))


def warp_loss(inpainting_network, driving, generated):
    """model.py 208-217. Compare the decoder's warped encoder maps to a fresh encode of
    the DRIVING image, occluded the same way:
        L = sum_i |get_encode(driving)[i] - generated['warped_encoder_maps'][-i-1]|.mean()
    `generated` is the inpainting forward output (must include 'occlusion_map' +
    'warped_encoder_maps'). driving: NHWC. Returns scalar mx.array.
    """
    occlusion_map = generated["occlusion_map"]
    encode_map = inpainting_network.get_encode(driving, occlusion_map)
    decode_map = generated["warped_encoder_maps"]
    value = mx.array(0.0)
    for i in range(len(encode_map)):
        value = value + mx.mean(mx.abs(encode_map[i] - decode_map[-i - 1]))
    return value


def kp_distance_loss(kp_driving):
    """MobilePortrait Δ3 kp_distance (model.py 235-241).

    Pulls the fused mixed keypoints (fg_kp) toward the frozen 106-pt FK landmarks
    (fk_kp) so the learned keypoints stay anchored to real facial structure. Compares
    the first n = min(num_fk, num_mixed) points (the two sets differ in count). Requires
    the kp extractor to be a MixedKPDetector (it produces fk_kp); returns the raw mean
    so the caller applies the loss weight, matching equivariance_loss / warp_loss.
    """
    fk = kp_driving["fk_kp"]
    mk = kp_driving["fg_kp"]
    n = min(fk.shape[1], mk.shape[1])
    return mx.mean(mx.abs(mk[:, :n] - fk[:, :n]))
