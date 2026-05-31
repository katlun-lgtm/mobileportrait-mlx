"""MLX port of src/modules/avd_network.py::AVDNetwork (Animation via Disentanglement).

Three MLPs of the form [Linear, BatchNorm1d, ReLU] x3 then Linear:
  id_encoder, pose_encoder: input_size(=10*num_tps) -> 256 -> 512 -> 1024 -> bottle
  decoder:                  (id_bottle+pose_bottle) -> 1024 -> 512 -> 256 -> input_size
Disentangles a source identity from a random pose, recombines into reconstructed kp.
Used by the separate AVD animation mode (relative motion), not the main reenactment path.

Each MLP stores 4 Linears (self.lins) + 3 BatchNorms (self.bns); ReLU is functional.
The torch nn.Sequential indices map: seq idx 3k -> lins[k], seq idx 3k+1 -> bns[k]
(0->lin0, 1->bn0, 3->lin1, 4->bn1, 6->lin2, 7->bn2, 9->lin3). load_avd_from_torch
applies that remap so a torch state_dict transfers key-for-key.
"""

import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class _MLP(nn.Module):
    """[Linear, BN, ReLU] x3 then Linear, over the given dims (len 5: in,h1,h2,h3,out)."""

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 5, dims
        self.lins = [nn.Linear(dims[i], dims[i + 1]) for i in range(4)]
        self.bns = [nn.BatchNorm(dims[i + 1]) for i in range(3)]

    def __call__(self, x):
        for k in range(3):
            x = nn.relu(self.bns[k](self.lins[k](x)))
        return self.lins[3](x)


class AVDNetwork(nn.Module):
    def __init__(self, num_tps, id_bottle_size=64, pose_bottle_size=64):
        super().__init__()
        self.num_tps = num_tps
        input_size = 5 * 2 * num_tps
        self.id_encoder = _MLP([input_size, 256, 512, 1024, id_bottle_size])
        self.pose_encoder = _MLP([input_size, 256, 512, 1024, pose_bottle_size])
        self.decoder = _MLP(
            [pose_bottle_size + id_bottle_size, 1024, 512, 256, input_size]
        )

    def __call__(self, kp_source, kp_random):
        bs = kp_source["fg_kp"].shape[0]
        pose_emb = self.pose_encoder(kp_random["fg_kp"].reshape(bs, -1))
        id_emb = self.id_encoder(kp_source["fg_kp"].reshape(bs, -1))
        rec = self.decoder(mx.concatenate([pose_emb, id_emb], axis=1))
        return {"fg_kp": rec.reshape(bs, self.num_tps * 5, -1)}


def load_avd_from_torch(mlx_model, torch_state_dict):
    """Transfer a torch AVDNetwork state_dict into the MLX model.

    torch keys: '{id_encoder,pose_encoder,decoder}.{0,1,3,4,6,7,9}.*' (Sequential idx).
    Remap each Sequential index to lins[k]/bns[k] (3k->lin k, 3k+1->bn k). All Linear/BN
    params are 1-D/2-D so no conv transpose; num_batches_tracked dropped.
    """
    flat = []
    for k, v in torch_state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        m = re.match(r"((?:id_encoder|pose_encoder|decoder)\.)(\d+)(\..*)$", k)
        if m:
            seq = int(m.group(2))
            sub = f"lins.{seq // 3}" if seq % 3 == 0 else f"bns.{seq // 3}"
            k = f"{m.group(1)}{sub}{m.group(3)}"
        flat.append((k, mx.array(v.detach().cpu().numpy())))
    mlx_keys = set(k for k, _ in tree_flatten(mlx_model.parameters()))
    torch_keys = set(k for k, _ in flat)
    mlx_model.unfreeze(recurse=True)
    mlx_model.update(tree_unflatten(flat))
    mlx_model.eval()
    return sorted(mlx_keys - torch_keys), sorted(torch_keys - mlx_keys)
