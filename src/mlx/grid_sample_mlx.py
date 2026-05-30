"""Differentiable 2D grid_sample for MLX — forward + backward (VJP) custom Metal kernels.

WHY: training warping models (TPS / FOMM / MobilePortrait) on Apple Silicon needs grid_sample's
BACKWARD on the GPU. PyTorch MPS lacks `grid_sampler_2d_backward` (falls back to CPU, ~1-core
serialized → the M3 Max GPU sits idle). MLX lets us write the backward as a custom Metal kernel and
wire it into autograd via mx.custom_function/.vjp, so fwd+bwd both run on the GPU.

The forward kernel here matches the channels-last forward already in ~/lp-mlx/kernels/
mlx_grid_sample.py (PyTorch `torch.nn.functional.grid_sample` semantics, mode='bilinear'):
  x (N,H,W,C), grid (N,Ho,Wo,2) [last dim (x,y) → (W,H)] -> out (N,Ho,Wo,C)
  align_corners ∈ {False,True}; padding_mode ∈ {'zeros','border'} ('reflection' not implemented).

The backward adds:
  - d_x : scatter-add of the same bilinear weights → needs atomic adds (atomic_outputs=True,
          init_value=0) because many output samples map to one input pixel.
  - d_grid : analytic gradient of the bilinear weights w.r.t. ix,iy, chained through the
          coord-unnormalization factor; summed over channels. Plain (non-atomic) output: each
          (n,ho,wo) grid element is written by exactly one thread.

────────────────────────────────────────────────────────────────────────────────────────────
⚠️ MLX VERSION: developed + validated against **mlx 0.31.2** (the version in ~/lp-mlx/.venv as of
2026-05-30). mx.fast.metal_kernel / mx.custom_function are evolving APIs — the kernel-call kwargs
(init_value, atomic_outputs) and the .vjp(primals, cotangent, output) signature are 0.31.x.
If MLX is upgraded and this breaks, pin back: `pip install mlx==0.31.2`. Keep this note current.
────────────────────────────────────────────────────────────────────────────────────────────
"""

import mlx.core as mx

# version this module was written/validated against (informational; see test for the assert)
VALIDATED_MLX_VERSION = "0.31.2"

_FWD = {}  # (align_corners, padding_mode) -> kernel
_DX = {}  # (align_corners, padding_mode) -> kernel  (atomic scatter into d_x)
_DGRID = {}  # (align_corners, padding_mode) -> kernel


def _unnorm(axis_var, dim_var, align_corners):
    """ix from normalized grid value, PyTorch semantics. Returns the C expr (a float)."""
    if align_corners:
        return f"(({axis_var} + 1.0f) * 0.5f * (float)({dim_var} - 1))"
    return f"((({axis_var} + 1.0f) * (float){dim_var} - 1.0f) * 0.5f)"


def _dunnorm(dim_var, align_corners):
    """d(ix)/d(gx): constant factor from the unnormalization above."""
    if align_corners:
        return f"(0.5f * (float)({dim_var} - 1))"
    return f"(0.5f * (float){dim_var})"


# ---------------------------------------------------------------- forward (channels-last 2D)
_FWD_SRC = """
    uint elem = thread_position_in_grid.x;
    int Hi = x_shape[1]; int Wi = x_shape[2]; int C = x_shape[3];
    int Ho = grid_shape[1]; int Wo = grid_shape[2];
    int w_stride = C; int h_stride = Wi * C; int n_stride = Hi * Wi * C;

    int c = (int)(elem % (uint)C); int tmp = (int)(elem / (uint)C);
    int wo = tmp % Wo; tmp /= Wo; int ho = tmp % Ho; tmp /= Ho; int n = tmp;

    int gbase = ((n * Ho + ho) * Wo + wo) * 2;
    float gx = (float)grid[gbase + 0]; float gy = (float)grid[gbase + 1];
    float ix = __IX__; float iy = __IY__;

    int x0 = (int)floor(ix); int x1 = x0 + 1;
    int y0 = (int)floor(iy); int y1 = y0 + 1;
    float wx1 = ix - (float)x0; float wx0 = 1.0f - wx1;
    float wy1 = iy - (float)y0; float wy0 = 1.0f - wy1;
    int xs[2] = {x0, x1}; int ys[2] = {y0, y1};
    float wxs[2] = {wx0, wx1}; float wys[2] = {wy0, wy1};

    int base = n * n_stride + c;
    float acc = 0.0f;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        int xi = xs[i]; int yi = ys[j];
        float w = wxs[i] * wys[j];
        int xc = min(max(xi, 0), Wi - 1); int yc = min(max(yi, 0), Hi - 1);
        float m = __MASK__;
        uint idx = (uint)(base + yc * h_stride + xc * w_stride);
        acc += w * m * (float)x[idx];
    }
    out[elem] = (T)acc;
"""

# ---------------------------------------------------------------- backward w.r.t. x (atomic scatter)
# one thread per output element (n,ho,wo,c); scatter the 4 weighted corner contributions of
# cot[elem] into d_x. atomic because corners from different output samples collide on input pixels.
_DX_SRC = """
    uint elem = thread_position_in_grid.x;
    int Hi = x_shape[1]; int Wi = x_shape[2]; int C = x_shape[3];
    int Ho = grid_shape[1]; int Wo = grid_shape[2];
    int w_stride = C; int h_stride = Wi * C; int n_stride = Hi * Wi * C;

    int c = (int)(elem % (uint)C); int tmp = (int)(elem / (uint)C);
    int wo = tmp % Wo; tmp /= Wo; int ho = tmp % Ho; tmp /= Ho; int n = tmp;

    int gbase = ((n * Ho + ho) * Wo + wo) * 2;
    float gx = (float)grid[gbase + 0]; float gy = (float)grid[gbase + 1];
    float ix = __IX__; float iy = __IY__;

    int x0 = (int)floor(ix); int x1 = x0 + 1;
    int y0 = (int)floor(iy); int y1 = y0 + 1;
    float wx1 = ix - (float)x0; float wx0 = 1.0f - wx1;
    float wy1 = iy - (float)y0; float wy0 = 1.0f - wy1;
    int xs[2] = {x0, x1}; int ys[2] = {y0, y1};
    float wxs[2] = {wx0, wx1}; float wys[2] = {wy0, wy1};

    float g = (float)cot[elem];
    int base = n * n_stride + c;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        int xi = xs[i]; int yi = ys[j];
        float w = wxs[i] * wys[j];
        int xc = min(max(xi, 0), Wi - 1); int yc = min(max(yi, 0), Hi - 1);
        float m = __MASK__;
        uint idx = (uint)(base + yc * h_stride + xc * w_stride);
        atomic_fetch_add_explicit(&d_x[idx], (T)(w * m * g), memory_order_relaxed);
    }
"""

# ---------------------------------------------------------------- backward w.r.t. grid
# one thread per (n,ho,wo); loop channels. d out/d ix = sum_corners (dw/dix) * m * x ; likewise iy.
# then chain by d ix/d gx (unnormalization factor). Non-atomic: unique writer per element.
_DGRID_SRC = """
    uint e = thread_position_in_grid.x;   // over N*Ho*Wo
    int Hi = x_shape[1]; int Wi = x_shape[2]; int C = x_shape[3];
    int Ho = grid_shape[1]; int Wo = grid_shape[2];
    int w_stride = C; int h_stride = Wi * C; int n_stride = Hi * Wi * C;

    int wo = (int)(e % (uint)Wo); int tmp = (int)(e / (uint)Wo);
    int ho = tmp % Ho; tmp /= Ho; int n = tmp;

    int gbase = ((n * Ho + ho) * Wo + wo) * 2;
    float gx = (float)grid[gbase + 0]; float gy = (float)grid[gbase + 1];
    float ix = __IX__; float iy = __IY__;

    int x0 = (int)floor(ix); int x1 = x0 + 1;
    int y0 = (int)floor(iy); int y1 = y0 + 1;
    float wx1 = ix - (float)x0; float wx0 = 1.0f - wx1;
    float wy1 = iy - (float)y0; float wy0 = 1.0f - wy1;
    // d weight / d ix and / d iy  (per corner: x-weight pairs with +-1 in ix)
    // corner (i,j): wx = (i? wx1 : wx0), wy = (j? wy1 : wy0)
    // dwx/dix: i? +1 : -1 ;  dwy/diy: j? +1 : -1
    int xs[2] = {x0, x1}; int ys[2] = {y0, y1};
    float wxs[2] = {wx0, wx1}; float wys[2] = {wy0, wy1};
    float dwx[2] = {-1.0f, 1.0f}; float dwy[2] = {-1.0f, 1.0f};

    int cobase = ((n * Ho + ho) * Wo + wo) * C;  // cot is (N,Ho,Wo,C), row-contiguous
    int base = n * n_stride;
    float gix = 0.0f; float giy = 0.0f;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        int xi = xs[i]; int yi = ys[j];
        int xc = min(max(xi, 0), Wi - 1); int yc = min(max(yi, 0), Hi - 1);
        float m = __MASK__;
        float wx_dix = dwx[i] * wys[j] * m;   // d(w*m)/dix for this corner
        float wy_diy = wxs[i] * dwy[j] * m;   // d(w*m)/diy for this corner
        uint xidx = (uint)(base + yc * h_stride + xc * w_stride);
        for (int c = 0; c < C; ++c) {
            float xv = (float)x[xidx + c];
            float gv = (float)cot[cobase + c];
            gix += gv * wx_dix * xv;
            giy += gv * wy_diy * xv;
        }
    }
    d_grid[gbase + 0] = (T)(gix * __DIX__);
    d_grid[gbase + 1] = (T)(giy * __DIY__);
"""


def _mask_expr(padding_mode):
    if padding_mode == "border":
        return "1.0f"
    return "((xi >= 0 && xi < Wi && yi >= 0 && yi < Hi) ? 1.0f : 0.0f)"


def _subst(src, align_corners, padding_mode, with_dgrad=False):
    s = src.replace("__IX__", _unnorm("gx", "Wi", align_corners))
    s = s.replace("__IY__", _unnorm("gy", "Hi", align_corners))
    s = s.replace("__MASK__", _mask_expr(padding_mode))
    if with_dgrad:
        s = s.replace("__DIX__", _dunnorm("Wi", align_corners))
        s = s.replace("__DIY__", _dunnorm("Hi", align_corners))
    return s


def _fwd_kernel(ac, pm):
    k = (ac, pm)
    if k not in _FWD:
        _FWD[k] = mx.fast.metal_kernel(
            name=f"gs2d_fwd_{int(ac)}_{pm}",
            input_names=["x", "grid"],
            output_names=["out"],
            source=_subst(_FWD_SRC, ac, pm),
            ensure_row_contiguous=True,
        )
    return _FWD[k]


def _dx_kernel(ac, pm):
    k = (ac, pm)
    if k not in _DX:
        _DX[k] = mx.fast.metal_kernel(
            name=f"gs2d_dx_{int(ac)}_{pm}",
            # x is passed so MLX generates x_shape (we use its dims, not its data).
            input_names=["cot", "x", "grid"],
            output_names=["d_x"],
            source=_subst(_DX_SRC, ac, pm),
            ensure_row_contiguous=True,
            atomic_outputs=True,
        )
    return _DX[k]


def _dgrid_kernel(ac, pm):
    k = (ac, pm)
    if k not in _DGRID:
        _DGRID[k] = mx.fast.metal_kernel(
            name=f"gs2d_dgrid_{int(ac)}_{pm}",
            input_names=["cot", "x", "grid"],
            output_names=["d_grid"],
            source=_subst(_DGRID_SRC, ac, pm, with_dgrad=True),
            ensure_row_contiguous=True,
        )
    return _DGRID[k]


def make_grid_sample_2d(align_corners=False, padding_mode="zeros"):
    """Return a differentiable channels-last 2D grid_sample(x,grid) bound to these options.

    x:(N,H,W,C) grid:(N,Ho,Wo,2) -> (N,Ho,Wo,C). Differentiable in BOTH x and grid via custom Metal
    forward+backward kernels (mx.custom_function/.vjp). Use under mx.grad/value_and_grad.
    """
    if padding_mode not in ("zeros", "border"):
        raise ValueError("padding_mode must be 'zeros' or 'border'")

    @mx.custom_function
    def grid_sample(x, grid):
        if grid.dtype != x.dtype:
            grid = grid.astype(x.dtype)
        N, _, _, C = x.shape
        out_shape = (N, grid.shape[1], grid.shape[2], C)
        total = N * grid.shape[1] * grid.shape[2] * C
        (out,) = _fwd_kernel(align_corners, padding_mode)(
            inputs=[x, grid],
            template=[("T", x.dtype)],
            output_shapes=[out_shape],
            output_dtypes=[x.dtype],
            grid=(total, 1, 1),
            threadgroup=(256, 1, 1),
        )
        return out

    @grid_sample.vjp
    def grid_sample_vjp(primals, cotangent, output):
        x, grid = primals
        if grid.dtype != x.dtype:
            grid = grid.astype(x.dtype)
        cot = cotangent.astype(x.dtype)
        N, Ho, Wo, C = cot.shape
        total = N * Ho * Wo * C
        # d_x: atomic scatter, init to zero. x passed only so x_shape exists in the kernel.
        (d_x,) = _dx_kernel(align_corners, padding_mode)(
            inputs=[cot, x, grid],
            template=[("T", x.dtype)],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            init_value=0.0,
            grid=(total, 1, 1),
            threadgroup=(256, 1, 1),
        )
        # d_grid: one thread per (n,ho,wo)
        ng = N * Ho * Wo
        (d_grid,) = _dgrid_kernel(align_corners, padding_mode)(
            inputs=[cot, x, grid],
            template=[("T", x.dtype)],
            output_shapes=[grid.shape],
            output_dtypes=[grid.dtype],
            grid=(ng, 1, 1),
            threadgroup=(256, 1, 1),
        )
        return d_x, d_grid

    return grid_sample
