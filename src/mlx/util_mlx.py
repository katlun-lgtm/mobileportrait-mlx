"""MLX port of the motion-math core from src/modules/util.py (the TPS reference fork).

Ported (each validated vs the PyTorch reference in tests/test_util_mlx_parity.py):
  - make_coordinate_grid
  - to_homogeneous / from_homogeneous
  - kp2gaussian
  - TPS (mode 'kp' = per-frame warp used at inference + main training warp;
         mode 'random' = train-only equivariance aug, noise injectable for parity)

NHWC / numpy-style broadcasting throughout. The conv blocks (ResBlock2d, Up/Down/
SameBlock2d, Encoder/Decoder/Hourglass, AntiAliasInterpolation2d) are a separate
mechanical port (blocks_mlx.py) — kept out of this file deliberately.
"""

import mlx.core as mx


def make_coordinate_grid(spatial_size, dtype=mx.float32):
    """meshgrid in [-1,1]^2, shape (h, w, 2), last axis = (x, y). Matches torch ref."""
    h, w = spatial_size
    x = mx.arange(w).astype(dtype)
    y = mx.arange(h).astype(dtype)
    x = 2.0 * (x / (w - 1)) - 1.0
    y = 2.0 * (y / (h - 1)) - 1.0
    xx = mx.broadcast_to(x.reshape(1, w), (h, w))
    yy = mx.broadcast_to(y.reshape(h, 1), (h, w))
    return mx.concatenate([xx[..., None], yy[..., None]], axis=2)


def to_homogeneous(coordinates):
    ones = mx.ones(list(coordinates.shape[:-1]) + [1], dtype=coordinates.dtype)
    return mx.concatenate([coordinates, ones], axis=-1)


def from_homogeneous(coordinates):
    return coordinates[..., :2] / coordinates[..., 2:3]


def kp2gaussian(kp, spatial_size, kp_variance):
    """Gaussian heatmap per keypoint. kp shape (..., 2) -> out (..., h, w)."""
    grid = make_coordinate_grid(spatial_size, kp.dtype)  # (h, w, 2)
    nld = len(kp.shape) - 1
    grid = grid.reshape((1,) * nld + grid.shape)  # (1..,h,w,2)
    kp = kp.reshape(kp.shape[:nld] + (1, 1, 2))  # (..,1,1,2)
    mean_sub = grid - kp  # broadcast -> (..,h,w,2)
    return mx.exp(-0.5 * mx.sum(mean_sub * mean_sub, axis=-1) / kp_variance)


def _batched_inv(L):
    """Inverse over leading batch dims. MLX inv runs on the CPU stream; flatten to 3D."""
    *batch, m, _ = L.shape
    flat = L.reshape((-1, m, m))
    inv = mx.linalg.inv(flat, stream=mx.cpu)
    return inv.reshape(tuple(batch) + (m, m))


@mx.custom_function
def _batched_solve(L, Y):
    """X = inv(L) @ Y over leading batch dims, with a hand-written VJP.

    MLX 0.32 has no vjp for `Inverse`, so a plain `_batched_inv(L) @ Y` cannot be
    backpropagated. This wraps the solve in a custom_function whose backward uses the
    analytic gradient of X = L^-1 Y — which only needs FORWARD inverses (those work):
        dY = (L^-1)^T @ G
        dL = -(dY @ X^T)
    Forward output is identical to `_batched_inv(L) @ Y`, so forward parity is preserved.
    """
    return _batched_inv(L) @ Y


@_batched_solve.vjp
def _batched_solve_vjp(primals, cotangent, output):
    L, _Y = primals
    G = cotangent
    X = output
    iLT = mx.swapaxes(_batched_inv(L), -1, -2)  # (L^-1)^T
    dY = iLT @ G
    dL = -(dY @ mx.swapaxes(X, -1, -2))
    return dL, dY


class TPS:
    """Thin-plate-spline transform. Faithful port of src/modules/util.py::TPS.

    mode 'kp':     keypoint-driven warp (Eq.2). kwargs kp_1, kp_2 each (bs, gs, n, 2).
    mode 'random': equivariance aug. For parity testing pass noise/control_params/
                   control_points explicitly; otherwise sampled with mx.random.
    """

    def __init__(self, mode, bs, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == "random":
            sigma_affine = kwargs["sigma_affine"]
            sigma_tps = kwargs["sigma_tps"]
            points_tps = kwargs["points_tps"]
            noise = kwargs.get("_noise")
            if noise is None:
                noise = mx.random.normal((bs, 2, 3)) * sigma_affine
            self.theta = noise + mx.eye(2, 3).reshape(1, 2, 3)
            cp = make_coordinate_grid((points_tps, points_tps))
            self.control_points = cp.reshape(1, points_tps * points_tps, 2)
            params = kwargs.get("_control_params")
            if params is None:
                params = mx.random.normal((bs, 1, points_tps**2)) * sigma_tps
            self.control_params = params
        elif mode == "kp":
            kp_1 = kwargs["kp_1"]
            kp_2 = kwargs["kp_2"]
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            # K = ||kp_1_i - kp_1_j||^2  (bs, gs, n, n)
            # NOTE: torch ref does norm()**2; we compute the squared distance directly.
            # The old `sqrt(sum(diff^2)); K=K*K` round-trip is identical in the forward but
            # its sqrt has an INFINITE gradient at the zero diagonal (kp_i - kp_i = 0) ->
            # NaN in backward. sum(diff^2) gives the same value with a finite gradient.
            diff = kp_1[:, :, :, None, :] - kp_1[:, :, None, :, :]
            K = mx.sum(diff * diff, axis=4)
            K = K * mx.log(K + 1e-9)

            one1 = mx.ones((bs, kp_1.shape[1], kp_1.shape[2], 1), dtype=kp_1.dtype)
            kp_1p = mx.concatenate([kp_1, one1], axis=3)  # (bs,gs,n,3)

            zero = mx.zeros((bs, kp_1.shape[1], 3, 3), dtype=kp_1.dtype)
            P = mx.concatenate([kp_1p, zero], axis=2)  # (bs,gs,n+3,3)
            L = mx.concatenate(
                [K, mx.transpose(kp_1p, (0, 1, 3, 2))], axis=2
            )  # (bs,gs,n+3,n)
            L = mx.concatenate([L, P], axis=3)  # (bs,gs,n+3,n+3)

            zero2 = mx.zeros((bs, kp_1.shape[1], 3, 2), dtype=kp_1.dtype)
            Y = mx.concatenate([kp_2, zero2], axis=2)  # (bs,gs,n+3,2)
            m = L.shape[2]
            eye = mx.eye(m).reshape((1, 1, m, m)) * 0.01
            L = L + eye

            param = _batched_solve(
                L, Y
            )  # (bs,gs,n+3,2); custom-vjp solve (inv has no vjp)
            self.theta = mx.transpose(param[:, :, n:, :], (0, 1, 3, 2))  # (bs,gs,2,3)
            self.control_points = kp_1
            self.control_params = param[:, :, :n, :]  # (bs,gs,n,2)
        else:
            raise ValueError("Error TPS mode")

    def transform_frame(self, frame):
        h, w = frame.shape[2], frame.shape[3]
        grid = make_coordinate_grid((h, w), frame.dtype).reshape(1, h * w, 2)
        shape = [self.bs, h, w, 2]
        if self.mode == "kp":
            shape.insert(1, self.gs)
        return self.warp_coordinates(grid).reshape(shape)

    def warp_coordinates(self, coordinates):
        theta = self.theta.astype(coordinates.dtype)
        control_points = self.control_points.astype(coordinates.dtype)
        control_params = self.control_params.astype(coordinates.dtype)

        if self.mode == "kp":
            # theta (bs,gs,2,3); coordinates (1,HW,2)
            transformed = (
                theta[:, :, :, :2] @ mx.transpose(coordinates, (0, 2, 1))
                + theta[:, :, :, 2:]
            )  # (bs,gs,2,HW)
            coords_v = coordinates.reshape(coordinates.shape[0], 1, 1, -1, 2)
            cp_v = control_points.reshape(self.bs, control_points.shape[1], -1, 1, 2)
            distances = coords_v - cp_v  # (bs,gs,n,HW,2)
            distances = distances * distances
            result = mx.sum(distances, axis=-1)  # (bs,gs,n,HW)
            result = result * mx.log(result + 1e-9)
            result = mx.transpose(result, (0, 1, 3, 2)) @ control_params  # (bs,gs,HW,2)
            transformed = (
                mx.transpose(transformed, (0, 1, 3, 2)) + result
            )  # (bs,gs,HW,2)
        elif self.mode == "random":
            theta = theta[:, None]  # (bs,1,2,3)
            transformed = (
                theta[:, :, :, :2] @ coordinates[..., None] + theta[:, :, :, 2:]
            )
            transformed = transformed[..., 0]
            ances = coordinates.reshape(
                coordinates.shape[0], -1, 1, 2
            ) - self.control_points.reshape(1, 1, -1, 2)
            distances = ances * ances
            result = mx.sum(distances, axis=-1)
            result = result * mx.log(result + 1e-9)
            result = result * control_params
            result = mx.sum(result, axis=2).reshape(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        else:
            raise ValueError("Error TPS mode")
        return transformed
