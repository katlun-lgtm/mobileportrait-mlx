"""Conv-only PROXY bench v2 — Core AI with EXPLICIT compute units.

Converts the conv U-Net once, then times Core AI pinned to GPU / Neural-Engine / CPU
via SpecializationOptions, plus torch-MPS (GPU) for cross-reference. Per-unit try/except
so the ANE 'program load failure' seen in v1 doesn't kill the GPU/CPU numbers.
"""

import time
import statistics
import shutil
import asyncio
import traceback
from pathlib import Path
import torch

N = 60


def cr(i, o, k=3, s=1, p=1):
    return torch.nn.Sequential(torch.nn.Conv2d(i, o, k, s, p), torch.nn.ReLU())


class ConvUNet(torch.nn.Module):
    def __init__(self, bx=64, maxf=512, ndown=3, in_ch=7, out_ch=3, nres=4):
        super().__init__()
        self.first = cr(in_ch, bx, 7, 1, 3)
        downs, ch = [], bx
        for _ in range(ndown):
            o = min(ch * 2, maxf)
            downs.append(cr(ch, o, 4, 2, 1))
            ch = o
        self.downs = torch.nn.ModuleList(downs)
        self.res = torch.nn.ModuleList([cr(ch, ch) for _ in range(nres)])
        ups = []
        for _ in range(ndown):
            o = max(ch // 2, bx)
            ups.append(
                torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode="nearest"), cr(ch, o)
                )
            )
            ch = o
        self.ups = torch.nn.ModuleList(ups)
        self.final = torch.nn.Conv2d(ch, out_ch, 7, 1, 3)

    def forward(self, x):
        x = self.first(x)
        for d in self.downs:
            x = d(x)
        for r in self.res:
            x = x + r(x)
        for u in self.ups:
            x = u(x)
        return self.final(x)


def summ(t):
    t = sorted(t)
    return statistics.median(t), t[len(t) // 10], t[(len(t) * 9) // 10]


def main():
    print(f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}")
    model = ConvUNet().half().eval()
    print(
        f"ConvUNet params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  in=1x7x256x256 fp16"
    )
    x = torch.randn(1, 7, 256, 256, dtype=torch.float16)

    # torch MPS fp16 (GPU reference)
    try:
        m = model.to("mps")
        xx = x.to("mps")
        with torch.no_grad():
            for _ in range(3):
                m(xx)
            torch.mps.synchronize()
            t = []
            for _ in range(N):
                s = time.perf_counter()
                m(xx)
                torch.mps.synchronize()
                t.append((time.perf_counter() - s) * 1000)
        md, p10, p90 = summ(t)
        print(f"[torch-MPS  GPU] median={md:.2f}ms  -> {1000 / md:.1f} fps")
    except Exception as e:
        print(f"[torch-MPS] FAIL {type(e).__name__}: {e}")
    model = model.to("cpu")

    # ---- Core AI: convert once, time each compute unit ----
    try:
        from coreai_torch import TorchConverter, get_decomp_table
        from coreai.runtime import NDArray, SpecializationOptions, ComputeUnitKind
    except Exception as e:
        print(f"[CoreAI] import FAIL {e}")
        return

    exp = torch.export.export(model, (x,)).run_decompositions(get_decomp_table())
    prog = (
        TorchConverter()
        .add_exported_program(
            exported_program=exp, input_names=["x"], output_names=["y"]
        )
        .to_coreai()
    )
    prog.optimize()
    print("[CoreAI] convert+optimize OK")

    asset_path = Path("/tmp/proxy.aimodel")
    if asset_path.exists():
        shutil.rmtree(asset_path)

    def opts_for(kind):
        if kind == "cpu":
            return SpecializationOptions.cpu_only()
        return SpecializationOptions.from_preferred_compute_unit_kind(
            getattr(ComputeUnitKind, kind)()
        )

    async def run_units():
        asset = prog.save_asset(asset_path)
        for kind in ["gpu", "neural_engine", "cpu"]:
            try:
                so = opts_for(kind)
                try:
                    cm = asset.executable(specialization_options=so)
                except TypeError:
                    cm = asset.executable(specialization_options=so._options)  # noqa: SLF001
                async with cm as ai:
                    fn = ai.load_function("main")
                    for _ in range(3):
                        await fn({"x": NDArray(data=x)})
                    t = []
                    for _ in range(N):
                        s = time.perf_counter()
                        await fn({"x": NDArray(data=x)})
                        t.append((time.perf_counter() - s) * 1000)
                md, p10, p90 = summ(t)
                print(
                    f"[CoreAI {kind:<13}] median={md:.2f}ms  p10={p10:.2f} p90={p90:.2f}  -> {1000 / md:.1f} fps"
                )
            except Exception as e:
                msg = str(e).splitlines()[0][:130]
                print(f"[CoreAI {kind:<13}] FAIL {type(e).__name__}: {msg}")

    asyncio.run(run_units())


if __name__ == "__main__":
    main()
