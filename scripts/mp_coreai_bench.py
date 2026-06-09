"""MobilePortrait → Core AI conversion + inference bench (M3 Max).

Two parts:
  PART 1 (runs on macOS 26.5 TODAY, torch only): build the torch inference model
    (dense_motion + inpainting, the same convnet our MLX forward_with_kp measures),
    torch.export it, run decompositions, and report the op profile — crucially whether
    grid_sampler / linalg solve survive export (the make-or-break ops for Core AI/ANE).
  PART 2 (needs macOS 27 / Xcode 27 + coreai-core + coreai-torch): convert the
    ExportedProgram to a Core AI AIProgram, optimize, run on Apple silicon, time it.

Apples-to-apples target = the MLX baseline measured 2026-06-09: forward_with_kp
median 44.25 ms (22.6 fps) on this same MBP. Core AI wins if it beats that on ANE
AND frees the GPU / cuts thermal.

Usage on MBP:
  source ~/lp-mlx/.venv/bin/activate
  python /tmp/mp_coreai_bench.py            # auto: export always; Core AI if SDK present
  python /tmp/mp_coreai_bench.py --part export   # force export-only dry run

Deps for PART 2 (install under macOS 27/Xcode 27): coreai-core==1.0.0b1 coreai-torch==0.4.0
"""

import os
import sys
import time
import shutil
import argparse
import statistics
import traceback
from pathlib import Path

REPO = os.path.expanduser("~/mobileportrait-mlx")
sys.path.insert(0, os.path.join(REPO, "src"))  # so `from modules.x import y` resolves
OUT = "/tmp/mp_coreai_bench.out"
MLX_BASELINE_MS = 44.25  # forward_with_kp median, measured 2026-06-09, same MBP


def log(m):
    with open(OUT, "a") as f:
        f.write(str(m) + "\n")
    print(m, flush=True)


def build_model():
    """Build dm+inpaint exactly as mp_train.py does (yaml-driven) so the architecture
    matches the trained model. Latency is weight-independent, so random init is fine."""
    import yaml
    from modules.dense_motion import DenseMotionNetwork
    from modules.inpainting_network import InpaintingNetwork

    cfg = yaml.safe_load(open(os.path.join(REPO, "reference-tps/config/vox-256.yaml")))
    common = cfg["model_params"]["common_params"]
    dm = DenseMotionNetwork(**common, **cfg["model_params"]["dense_motion_params"])
    inp = InpaintingNetwork(**cfg["model_params"]["generator_params"], **common)

    class _Wrap(torch.nn.Module):
        def __init__(self, dm, inp):
            super().__init__()
            self.dm, self.inp = dm, inp

        def forward(self, source, kp_source_v, kp_driving_v):
            dense = self.dm(
                source_image=source,
                kp_driving={"fg_kp": kp_driving_v},
                kp_source={"fg_kp": kp_source_v},
                bg_param=None,
                dropout_flag=False,
                dropout_p=0,
            )
            return self.inp(source, dense)["prediction"]  # inference: no mv/pseudo_bg

    return _Wrap(dm, inp).eval(), int(common["num_tps"])


def make_inputs(num_tps):
    import torch

    K = num_tps * 5
    g = torch.Generator().manual_seed(0)
    src = torch.randn(1, 3, 256, 256, generator=g)  # torch reference is NCHW
    kp_s = torch.randn(1, K, 2, generator=g)
    kp_d = torch.randn(1, K, 2, generator=g)
    return (src, kp_s, kp_d)


def do_export(model, ex):
    import torch

    log("torch.export.export ...")
    exported = torch.export.export(model, ex)
    try:
        from coreai_torch import get_decomp_table

        exported = exported.run_decompositions(get_decomp_table())
        tbl = "coreai_torch.get_decomp_table"
    except Exception as e:
        exported = exported.run_decompositions()  # torch default (26.5 dry run)
        tbl = f"torch-default ({type(e).__name__})"
    ops = sorted(
        {str(n.target) for n in exported.graph.nodes if n.op == "call_function"}
    )
    log(f"EXPORT_OK  decomp={tbl}  n_ops={len(ops)}")
    flags = [
        o
        for o in ops
        if any(
            k in o.lower()
            for k in (
                "grid_sampler",
                "grid_sample",
                "solve",
                "inverse",
                "linalg",
                "affine_grid",
            )
        )
    ]
    log("RISK_OPS (ANE-unfriendly / may force GPU/CPU fallback):")
    for o in flags:
        log(f"   • {o}")
    if not flags:
        log("   (none — fully decomposed to primitives; promising for ANE)")
    try:
        torch.export.save(exported, "/tmp/mp_exported.pt2")
        log("saved ExportedProgram -> /tmp/mp_exported.pt2")
    except Exception as e:
        log(f"(save skipped: {type(e).__name__})")
    return exported


def do_coreai(exported, ex, N=60):
    """PART 2 — needs Xcode 27 / coreai SDK. Convert, optimize, run on Apple silicon, time."""
    from coreai_torch import TorchConverter  # noqa: F401 (import = availability gate)

    log("Converting ExportedProgram -> Core AI AIProgram ...")
    conv = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["source", "kp_source_v", "kp_driving_v"],
        output_names=["prediction"],
    )
    prog = conv.to_coreai()
    prog.optimize()
    log("CONVERT_OK + optimize() done. Loading runtime ...")

    # Minimal standalone runtime (faithful to apple/coreai-models coreai_runner.py).
    import asyncio
    from contextlib import AsyncExitStack
    from coreai.runtime import NDArray

    asset = Path("/tmp/mp_coreai_asset.aimodel")
    if asset.exists():
        shutil.rmtree(asset)
    stack = AsyncExitStack()

    async def _load():
        a = prog.save_asset(asset)
        ai = await stack.enter_async_context(a.executable())
        return ai.load_function("main")

    fn = asyncio.run(_load())
    names = ["source", "kp_source_v", "kp_driving_v"]

    def forward():
        async def go():
            ins = {n: NDArray(data=t) for n, t in zip(names, ex)}
            return await fn(ins)

        return asyncio.run(go())

    forward()  # warmup / compile
    t = []
    for _ in range(N):
        s = time.perf_counter()
        forward()
        t.append((time.perf_counter() - s) * 1000.0)
    t.sort()
    med = statistics.median(t)
    log(
        f"[CoreAI] inference  median={med:.2f}ms  p10={t[len(t) // 10]:.2f}  p90={t[(len(t) * 9) // 10]:.2f}  -> {1000.0 / med:.1f} fps"
    )
    log(
        f"[MLX baseline] forward_with_kp median={MLX_BASELINE_MS:.2f}ms  -> {1000.0 / MLX_BASELINE_MS:.1f} fps"
    )
    verdict = "Core AI FASTER" if med < MLX_BASELINE_MS else "MLX faster"
    log(f"VERDICT: {verdict}  (Core AI {MLX_BASELINE_MS / med:.2f}x MLX)")
    asyncio.run(stack.aclose())


def main():
    global torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["export", "full"], default="full")
    args = ap.parse_args()
    open(OUT, "w").close()
    try:
        import torch  # noqa

        log(f"torch {torch.__version__}")
        model, num_tps = build_model()
        log(f"BUILD_OK num_tps={num_tps}")
        ex = make_inputs(num_tps)
        # sanity: eager forward works + shape
        with torch.no_grad():
            y = model(*ex)
        log(f"EAGER_OK prediction shape={tuple(y.shape)}")
        exported = do_export(model, ex)
        if args.part == "full":
            try:
                do_coreai(exported, ex)
            except ImportError as e:
                log(f"\nPART 2 (Core AI) SKIPPED — SDK not present: {e}")
                log(
                    "Install macOS 27 / Xcode 27, then: pip install coreai-core==1.0.0b1 coreai-torch==0.4.0"
                )
                log("Re-run: python /tmp/mp_coreai_bench.py")
        log("BENCH_END")
    except Exception:
        log("BENCH_FAIL\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
