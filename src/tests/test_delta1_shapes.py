import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from src.modules.keypoint_detector import MixedKPDetector

m = MixedKPDetector(num_tps=10, fk_backend="stub").train()
img = torch.rand(2, 3, 256, 256)
out = m(img)
shapes = {k: tuple(v.shape) for k, v in out.items()}
print("Δ1 output shapes:", shapes)
assert out['fg_kp'].shape == (2, 50, 2), out['fg_kp'].shape
assert out['nk_kp'].shape == (2, 50, 2)
assert out['fk_kp'].shape == (2, 106, 2)
assert float(out['fg_kp'].min()) >= -1.0 and float(out['fg_kp'].max()) <= 1.0, "fg_kp not in [-1,1]"

# gradient: flows through NK + MixedKP MLP; FK is frozen (no_grad)
out['fg_kp'].sum().backward()
nk_grads = sum(p.grad.abs().sum().item() for p in m.nk.parameters() if p.grad is not None)
mlp_grads = sum(p.grad.abs().sum().item() for p in m.mixed.parameters() if p.grad is not None)
fk_has_grad = any(p.grad is not None for p in m.fk.parameters())
print(f"NK grad sum {nk_grads:.3f} (>0 ✓), MixedKP grad sum {mlp_grads:.3f} (>0 ✓), FK has grad: {fk_has_grad} (False ✓)")
assert nk_grads > 0 and mlp_grads > 0 and not fk_has_grad
n_params = sum(p.numel() for p in m.parameters())
print(f"params: {n_params/1e6:.2f}M  | drop-in fg_kp matches KPDetector contract (bs,50,2)")
print("Δ1 SHAPE TEST PASS")
