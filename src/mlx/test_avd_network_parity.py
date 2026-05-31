"""Parity: src/mlx/avd_network_mlx.py vs torch AVDNetwork.

Loads identical torch weights (with randomised BatchNorm running stats so eval-mode BN is
exercised, not the trivial mean=0/var=1) and compares the reconstructed fg_kp on random
source + random-pose keypoints. Run on the Mac:
    cd ~/mobileportrait-mlx && ~/lp-mlx/.venv/bin/python src/mlx/test_avd_network_parity.py
"""

import importlib
import os
import sys

import numpy as np
import torch

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

ref = importlib.import_module("modules.avd_network")
from avd_network_mlx import AVDNetwork as MlxAVD  # noqa: E402
from avd_network_mlx import load_avd_from_torch  # noqa: E402

NUM_TPS = 10
BS = 2
TOL = 5e-3

torch.manual_seed(0)
np.random.seed(0)

tmodel = ref.AVDNetwork(num_tps=NUM_TPS).eval()
# randomise every BatchNorm's running stats + affine so eval-mode BN does real work
for m in tmodel.modules():
    if isinstance(m, torch.nn.BatchNorm1d):
        m.running_mean.normal_(0, 0.5)
        m.running_var.uniform_(0.5, 1.5)
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.normal_(0.0, 0.1)

mmodel = MlxAVD(num_tps=NUM_TPS)
miss, unexp = load_avd_from_torch(mmodel, tmodel.state_dict())
print(f"KEYDRIFT missing={len(miss)} unexpected={len(unexp)}")
if miss:
    print("  missing:", miss[:8])
if unexp:
    print("  unexpected:", unexp[:8])

src = np.random.rand(BS, NUM_TPS * 5, 2).astype("float32") * 2 - 1
rnd = np.random.rand(BS, NUM_TPS * 5, 2).astype("float32") * 2 - 1
with torch.no_grad():
    tout = tmodel({"fg_kp": torch.from_numpy(src)}, {"fg_kp": torch.from_numpy(rnd)})
    t = tout["fg_kp"].numpy()

mout = mmodel({"fg_kp": mx.array(src)}, {"fg_kp": mx.array(rnd)})
mx.eval(mout["fg_kp"])
m = np.array(mout["fg_kp"])

fails = 0
if t.shape != m.shape:
    print(f"FAIL shape torch={t.shape} mlx={m.shape}")
    fails += 1
else:
    md = float(np.abs(t - m).max())
    mn = float(np.abs(t - m).mean())
    ok = md < TOL
    print(
        f"{'PASS' if ok else 'FAIL'} fg_kp maxdiff={md:.3e} mean={mn:.3e} tol={TOL:.0e}"
    )
    if not ok:
        fails += 1

print(f"FAILS={fails}")
print("RESULT_ALL_PASS" if fails == 0 else "RESULT_FAIL")
