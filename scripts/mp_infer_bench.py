"""MobilePortrait MLX INFERENCE latency bench (M3 Max).

Measures the GPU forward that Core AI would accelerate: dense_motion + inpainting
(forward_with_kp) at 256px, plus the NK keypoint detector (resnet18). FK (insightface
ONNX) is identical under MLX or Core AI, so it is excluded from this comparison.

Reads nothing it didn't just compute. Writes results to /tmp/mp_infer_bench.out.
"""

import os
import sys
import time
import statistics
import traceback

OUT = "/tmp/mp_infer_bench.out"


def log(m):
    with open(OUT, "a") as f:
        f.write(str(m) + "\n")
    print(m, flush=True)


open(OUT, "w").close()
sys.path.insert(0, os.path.expanduser("~/mobileportrait-mlx/src/mlx"))

try:
    import numpy as np
    import mlx.core as mx
    from mixed_kp_mlx import MixedKPDetector
    from dense_motion_mlx import DenseMotionNetwork
    from inpainting_network_mlx import InpaintingNetwork
    from pipeline_mlx import MobilePortraitPipeline

    CKPT = os.path.expanduser("~/mobileportrait-mlx/checkpoints/real_best.safetensors")
    log(f"mlx {mx.__version__}  ckpt_exists={os.path.exists(CKPT)}")

    kp = MixedKPDetector(num_tps=10, fk_backend="stub")
    dm = DenseMotionNetwork(
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
    inp = InpaintingNetwork(
        num_channels=3,
        block_expansion=64,
        max_features=512,
        num_down_blocks=3,
        multi_mask=True,
    )
    model = MobilePortraitPipeline(kp, dm, inp)
    try:
        model.load_weights(CKPT)
        log("LOAD_WEIGHTS_OK (strict)")
    except Exception as e:
        model.load_weights(CKPT, strict=False)
        log(f"LOAD_WEIGHTS_OK (non-strict; {type(e).__name__})")
    model.eval()

    rng = np.random.default_rng(0)
    src = mx.array(rng.standard_normal((1, 256, 256, 3)).astype(np.float32))
    drv = mx.array(rng.standard_normal((1, 256, 256, 3)).astype(np.float32))
    kp_s = {"fg_kp": mx.array(rng.standard_normal((1, 50, 2)).astype(np.float32))}
    kp_d = {"fg_kp": mx.array(rng.standard_normal((1, 50, 2)).astype(np.float32))}

    N = 60

    # ---- (A) forward_with_kp: dense_motion + inpainting (the heavy convnet) ----
    out = model.forward_with_kp(src, kp_s, kp_d)  # warmup compile
    mx.eval(out["prediction"])
    log(f"prediction shape={out['prediction'].shape}")
    t = []
    for _ in range(N):
        s = time.perf_counter()
        o = model.forward_with_kp(src, kp_s, kp_d)
        mx.eval(o["prediction"])
        t.append((time.perf_counter() - s) * 1000.0)
    t.sort()
    med, p10, p90 = statistics.median(t), t[len(t) // 10], t[(len(t) * 9) // 10]
    log(
        f"[A] forward_with_kp (dm+inpaint)  median={med:.2f}ms  p10={p10:.2f}  p90={p90:.2f}  -> {1000.0 / med:.1f} fps"
    )

    # ---- (B) NK keypoint detector (resnet18), per frame ----
    try:
        _ = model.kp_extractor.nk(drv)["fg_kp"]
        mx.eval(_)
        tn = []
        for _ in range(N):
            s = time.perf_counter()
            r = model.kp_extractor.nk(drv)["fg_kp"]
            mx.eval(r)
            tn.append((time.perf_counter() - s) * 1000.0)
        tn.sort()
        log(f"[B] NK detector (resnet18)        median={statistics.median(tn):.2f}ms")
        full = med + statistics.median(tn)
        log(
            f"[A+B] MLX per driving-frame (excl FK ONNX)  ~{full:.2f}ms  -> {1000.0 / full:.1f} fps"
        )
    except Exception as e:
        log(f"[B] NK detector skipped: {type(e).__name__}: {e}")

    log("BENCH_END")
except Exception:
    log("BENCH_FAIL\n" + traceback.format_exc())
