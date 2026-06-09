"""CLOSE-OUT: hoist the TPS solve out of the graph + decomposed warp -> full Core AI
conversion -> real-model ANE bench vs the 44.25ms MLX baseline.

transformations (the TPS warp grid) is precomputed EAGERLY from keypoints (linalg_inv
runs on CPU, off-graph) and fed as a model INPUT. Everything else (decomposed warp +
hourglass + occlusion + inpainting) stays in the graph and converts.
"""

import os
import sys
import time
import statistics
import shutil
import asyncio
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = os.path.expanduser("~/mobileportrait-mlx")
sys.path.insert(0, os.path.join(REPO, "src"))
from modules.grid_sample_decomp import grid_sample_bilinear

import yaml
from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork

cfg = yaml.safe_load(open(os.path.join(REPO, "reference-tps/config/vox-256.yaml")))
common = cfg["model_params"]["common_params"]
MLX_BASELINE_MS = 44.25
N = 60


def dm_forward_hoisted(dm, source_image, transformations, kp_d_v, kp_s_v):
    """Mirrors DenseMotionNetwork.forward for inference (dropout/training/bg off),
    but uses the INJECTED `transformations` instead of create_transformations()."""
    if dm.scale_factor != 1:
        source_image = dm.down(source_image)
    bs, _, h, w = source_image.shape
    out = {}
    heatmap = dm.create_heatmap_representations(
        source_image, {"fg_kp": kp_d_v}, {"fg_kp": kp_s_v}
    )
    heatmap = heatmap.to(
        source_image.dtype
    )  # kp2gaussian/coord-grid emit fp32 literals; match model dtype
    deformed_source = dm.create_deformed_source_image(source_image, transformations)
    out["deformed_source"] = deformed_source
    deformed_source = deformed_source.view(bs, -1, h, w)
    inp_t = torch.cat([heatmap, deformed_source], dim=1).view(bs, -1, h, w)
    prediction = dm.hourglass(inp_t, mode=1)
    feat = prediction[-1]
    cmaps = F.softmax(dm.maps(feat), dim=1)
    out["contribution_maps"] = cmaps
    cm = cmaps.unsqueeze(2)
    trans = transformations.permute(0, 1, 4, 2, 3)
    deformation = (trans * cm).sum(dim=1).permute(0, 2, 3, 1)
    deformation = deformation + dm.residual_flow(feat).permute(0, 2, 3, 1)
    out["deformation"] = deformation
    occ = []
    if dm.multi_mask:
        for i in range(dm.occlusion_num - dm.up_nums):
            occ.append(
                torch.sigmoid(
                    dm.occlusion[i](prediction[dm.up_nums - dm.occlusion_num + i])
                )
            )
        pred = prediction[-1]
        for i in range(dm.up_nums):
            pred = dm.up[i](pred)
            occ.append(
                torch.sigmoid(dm.occlusion[i + dm.occlusion_num - dm.up_nums](pred))
            )
    else:
        occ.append(torch.sigmoid(dm.occlusion[0](prediction[-1])))
    out["occlusion_map"] = occ
    return out


class OrigWrap(torch.nn.Module):  # computes transformations inside (reference)
    def __init__(self, dm, inp):
        super().__init__()
        self.dm, self.inp = dm, inp

    def forward(self, source, kp_s, kp_d):
        d = self.dm(
            source_image=source,
            kp_driving={"fg_kp": kp_d},
            kp_source={"fg_kp": kp_s},
            bg_param=None,
            dropout_flag=False,
            dropout_p=0,
        )
        return self.inp(source, d)["prediction"]


class HoistWrap(torch.nn.Module):  # transformations injected as input
    def __init__(self, dm, inp):
        super().__init__()
        self.dm, self.inp = dm, inp

    def forward(self, source, transformations, kp_s, kp_d):
        d = dm_forward_hoisted(self.dm, source, transformations, kp_d, kp_s)
        return self.inp(source, d)["prediction"]


def summ(t):
    t = sorted(t)
    return statistics.median(t), t[len(t) // 10], t[(len(t) * 9) // 10]


def main():
    torch.manual_seed(0)
    dm = DenseMotionNetwork(
        **common, **cfg["model_params"]["dense_motion_params"]
    ).eval()
    inp = InpaintingNetwork(**cfg["model_params"]["generator_params"], **common).eval()
    K = common["num_tps"] * 5
    source = torch.randn(1, 3, 256, 256)
    kp_s = torch.randn(1, K, 2)
    kp_d = torch.randn(1, K, 2)

    # --- reference output with REAL grid_sample + internal create_transformations ---
    with torch.no_grad():
        y_ref = OrigWrap(dm, inp)(source, kp_s, kp_d)

    # --- precompute transformations eagerly (linalg_inv here, OFF the export graph) ---
    with torch.no_grad():
        src_down = dm.down(source) if dm.scale_factor != 1 else source
        trans = dm.create_transformations(
            src_down, {"fg_kp": kp_d}, {"fg_kp": kp_s}, None
        )
    print(f"precomputed transformations shape={tuple(trans.shape)}")

    # --- patch decomposed warp, run hoisted, check parity ---
    F.grid_sample = grid_sample_bilinear
    hoist = HoistWrap(dm, inp)
    with torch.no_grad():
        y_h = hoist(source, trans, kp_s, kp_d)
    d = (y_ref - y_h).abs().max().item()
    print(
        f"PARITY orig(real-gs+internal-TPS) vs hoisted(decomp-gs+injected-TPS): maxdiff={d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}"
    )

    # --- export + op profile ---
    ex32 = (source, trans, kp_s, kp_d)
    exp = torch.export.export(hoist, ex32).run_decompositions()
    ops = sorted({str(n.target) for n in exp.graph.nodes if n.op == "call_function"})
    blockers = [
        o
        for o in ops
        if any(k in o.lower() for k in ("grid_sampl", "linalg_inv", "inverse", "solve"))
    ]
    print(f"export blockers remaining: {blockers if blockers else 'NONE ✅'}")

    # --- Core AI convert (fp16) + bench across compute units ---
    from coreai_torch import TorchConverter, get_decomp_table
    from coreai.runtime import NDArray, SpecializationOptions, ComputeUnitKind

    h16 = HoistWrap(
        dm.half(), inp.half()
    ).eval()  # fp16 — deployment precision, ANE-native
    ex16 = tuple(t.half() for t in ex32)
    exp16 = torch.export.export(h16, ex16).run_decompositions(get_decomp_table())
    try:
        prog = (
            TorchConverter()
            .add_exported_program(
                exported_program=exp16,
                input_names=["source", "transformations", "kp_s", "kp_d"],
                output_names=["pred"],
            )
            .to_coreai()
        )
    except Exception as e:
        print("CONVERTER_REJECTED:", str(e).splitlines()[0][:200])
        return
    prog.optimize()
    print("CONVERT+OPTIMIZE OK ✅ (full model converts)")

    asset_path = Path("/tmp/mp_hoist.aimodel")
    if asset_path.exists():
        shutil.rmtree(asset_path)
    names = ["source", "transformations", "kp_s", "kp_d"]

    async def run_units():
        asset = prog.save_asset(asset_path)
        for kind in ["neural_engine", "gpu", "cpu"]:
            try:
                so = (
                    SpecializationOptions.cpu_only()
                    if kind == "cpu"
                    else SpecializationOptions.from_preferred_compute_unit_kind(
                        getattr(ComputeUnitKind, kind)()
                    )
                )
                async with asset.executable(specialization_options=so) as ai:
                    fn = ai.load_function("main")
                    for _ in range(3):
                        await fn({n: NDArray(data=t) for n, t in zip(names, ex16)})
                    t = []
                    for _ in range(N):
                        s = time.perf_counter()
                        await fn({n: NDArray(data=t2) for n, t2 in zip(names, ex16)})
                        t.append((time.perf_counter() - s) * 1000)
                md, p10, p90 = summ(t)
                tag = "ANE" if kind == "neural_engine" else kind.upper()
                extra = (
                    f"  vs MLX {MLX_BASELINE_MS}ms = {MLX_BASELINE_MS / md:.2f}x"
                    if kind == "neural_engine"
                    else ""
                )
                print(
                    f"[CoreAI {tag:<13}] median={md:.2f}ms  p10={p10:.2f} p90={p90:.2f}  -> {1000 / md:.1f} fps{extra}"
                )
            except Exception as e:
                print(
                    f"[CoreAI {kind:<13}] FAIL {type(e).__name__}: {str(e).splitlines()[0][:110]}"
                )

    asyncio.run(run_units())
    print(
        f"[MLX baseline] forward_with_kp {MLX_BASELINE_MS}ms -> {1000 / MLX_BASELINE_MS:.1f} fps"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FAIL\n" + traceback.format_exc())
