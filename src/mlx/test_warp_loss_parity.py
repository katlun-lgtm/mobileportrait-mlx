"""Parity: losses_mlx.warp_loss vs torch model.py warp_loss term.

Full pipeline both sides (identical transferred weights), feed a real torch dense_motion
dict to the MLX inpainting (so dm matches exactly), compare the warp_loss scalar.
Run on the Mac:
  cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_warp_loss_parity.py
"""

import os
import sys
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
import torch
import importlib

ref_dm = importlib.import_module("modules.dense_motion")
ref_inp = importlib.import_module("modules.inpainting_network")
import inpainting_network_mlx as MINP
import losses_mlx as ML

np.random.seed(0)
torch.manual_seed(0)
DM_CFG = dict(
    block_expansion=64,
    num_blocks=5,
    max_features=1024,
    num_tps=10,
    num_channels=3,
    scale_factor=0.25,
    bg=False,
    multi_mask=True,
    kp_variance=0.01,
)
INP_CFG = dict(
    num_channels=3,
    block_expansion=64,
    max_features=512,
    num_down_blocks=3,
    multi_mask=True,
)
BS = 2
fails = 0

# torch modules
t_dm = ref_dm.DenseMotionNetwork(**DM_CFG).eval()
t_inp = ref_inp.InpaintingNetwork(
    **INP_CFG
).train()  # train() so warped_encoder_maps branch active
m_inp = MINP.InpaintingNetwork(**INP_CFG)
MINP.load_inpaint_from_torch(m_inp, t_inp.state_dict())

src = np.random.randn(BS, 3, 256, 256).astype(np.float32)
kp_s = {"fg_kp": torch.tensor((np.random.rand(BS, 50, 2) * 2 - 1).astype(np.float32))}
kp_d = {"fg_kp": torch.tensor((np.random.rand(BS, 50, 2) * 2 - 1).astype(np.float32))}

with torch.no_grad():
    dm_t = t_dm(torch.tensor(src), kp_d, kp_s)
    gen_t = t_inp(torch.tensor(src), dm_t)
    # torch warp_loss (model.py 208-216)
    occ = gen_t["occlusion_map"]
    enc = t_inp.get_encode(
        torch.tensor(src), occ
    )  # NOTE: model.py uses x["driving"]; here driving=src for the test
    dec = gen_t["warped_encoder_maps"]
    tval = 0.0
    for i in range(len(enc)):
        tval += torch.abs(enc[i] - dec[-i - 1]).mean()
    t_loss = float(tval)

# mlx: feed the SAME torch dm dict (NHWC-converted)
dm_m = {
    "deformation": mx.array(dm_t["deformation"].numpy()),
    "contribution_maps": mx.array(
        dm_t["contribution_maps"].numpy().transpose(0, 2, 3, 1)
    ),
    "deformed_source": mx.array(
        dm_t["deformed_source"].numpy().transpose(0, 1, 3, 4, 2)
    ),
    "occlusion_map": [
        mx.array(o.numpy().transpose(0, 2, 3, 1)) for o in dm_t["occlusion_map"]
    ],
}
src_m = mx.array(src.transpose(0, 2, 3, 1))
gen_m = m_inp(src_m, dm_m)
m_loss = float(ML.warp_loss(m_inp, src_m, gen_m))

rel = abs(m_loss - t_loss) / (abs(t_loss) + 1e-9)
print(f"warp_loss mlx={m_loss:.6f} torch={t_loss:.6f} rel={rel:.3e}")
# also check warped_encoder_maps shapes match
print(
    f"n_warped_maps mlx={len(gen_m['warped_encoder_maps'])} torch={len(gen_t['warped_encoder_maps'])}"
)
ok = rel <= 5e-3
print(("PASS" if ok else "FAIL") + " warp_loss")
if not ok:
    fails += 1
print(f"\nFAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_ANY_FAIL")
