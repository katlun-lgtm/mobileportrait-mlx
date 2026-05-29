"""Warm-start loader — initialise MobilePortrait modules from a TPS `vox.pth.tar` checkpoint.

The Stage-B head start: the dense-motion Hourglass, the synthesis U-Net, and the NK detector
are all inherited from TPS verbatim; only the MobilePortrait delta layers train from scratch.

Three mismatches vs the pristine TPS checkpoint are handled here:

  1. kp_detector — our `MixedKPDetector` wraps TPS's `KPDetector` under `self.nk.*`, so the
     checkpoint's `fg_encoder.*` keys are remapped to `nk.fg_encoder.*`. The `mixed` MLP and the
     frozen `fk` detector have no checkpoint counterpart (fresh / not trained).
  2. inpainting `first` conv — grew from `num_channels` (3) to `num_channels+4` (7) input
     channels for the pseudo-BG + fg-mask (Δ4b). The checkpoint's 3-channel weight is copied
     into the first 3 input slots; the extra 4 channels start at zero (so an all-zero pseudo-BG
     input reproduces the original TPS behaviour exactly at init).
  3. fresh delta layers — `residual_flow`, `fg_mask_head`, `lmk_mask_head` (dense_motion) and
     `mv_merge` (inpainting) are absent from the checkpoint; loaded with strict=False.

Everything else must match exactly; we assert no *unexpected* keys remain and that the only
*missing* keys are the known delta layers, so a silent shape/name drift can't pass unnoticed.
"""

import torch


# Keys we expect to be absent from a TPS checkpoint (the MobilePortrait deltas + frozen FK).
_KP_FRESH_PREFIXES = ("mixed.", "fk.")
_DM_FRESH_PREFIXES = ("residual_flow.", "fg_mask_head.", "lmk_mask_head.")
_INP_FRESH_PREFIXES = ("mv_merge.",)


def _remap_kp_state(ckpt_kp_state):
    """TPS KPDetector keys (`fg_encoder.*`) -> MixedKPDetector keys (`nk.fg_encoder.*`)."""
    return {f"nk.{k}": v for k, v in ckpt_kp_state.items()}


def _expand_first_conv(model_state, ckpt_state, num_channels=3):
    """Copy a 3-in-channel `first.conv.weight` into the model's (num_channels+4)-in-channel one.

    Mutates `ckpt_state` in place so the expanded weight matches the model and load_state_dict
    accepts it. Returns True if an expansion was applied.
    """
    key = "first.conv.weight"
    if key not in ckpt_state or key not in model_state:
        return False
    ck_w = ckpt_state[key]  # (out, num_channels, k, k)
    md_w = model_state[key]  # (out, num_channels+4, k, k)
    if ck_w.shape == md_w.shape:
        return False  # no expansion needed (use_pseudo_bg=False)
    assert ck_w.shape[1] == num_channels, (
        f"unexpected checkpoint first-conv in-channels {ck_w.shape[1]} != {num_channels}"
    )
    new_w = torch.zeros_like(md_w)
    new_w[:, :num_channels] = ck_w
    ckpt_state[key] = new_w
    return True


def _load_relaxed(module, state, *, allowed_missing_prefixes, what):
    """load_state_dict(strict=False) with a guard: unexpected keys are fatal, and missing keys
    must all fall under `allowed_missing_prefixes` (the known fresh delta layers)."""
    result = module.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            f"[warmstart:{what}] unexpected checkpoint keys (name/shape drift?): "
            f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    bad_missing = [
        k
        for k in result.missing_keys
        if not k.startswith(tuple(allowed_missing_prefixes))
    ]
    if bad_missing:
        raise RuntimeError(
            f"[warmstart:{what}] unexpected MISSING keys (not a known delta "
            f"layer): {bad_missing[:8]}{'...' if len(bad_missing) > 8 else ''}"
        )
    fresh = [k for k in result.missing_keys]
    return fresh


def warm_start_from_tps(
    checkpoint_path,
    *,
    kp_detector=None,
    dense_motion_network=None,
    inpainting_network=None,
    num_channels=3,
    map_location="cpu",
    verbose=True,
):
    """Load a TPS `vox.pth.tar` into the (extended) MobilePortrait modules.

    Pass any subset of the three modules; each is warm-started independently. The MobilePortrait
    delta layers are left at their fresh init. Returns a dict of the fresh (un-loaded) param
    names per module, so the caller can verify exactly what starts from scratch.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    fresh = {}

    if kp_detector is not None:
        assert "kp_detector" in ckpt, "checkpoint missing 'kp_detector'"
        state = _remap_kp_state(ckpt["kp_detector"])
        fresh["kp_detector"] = _load_relaxed(
            kp_detector,
            state,
            allowed_missing_prefixes=_KP_FRESH_PREFIXES,
            what="kp_detector",
        )

    if dense_motion_network is not None:
        assert "dense_motion_network" in ckpt, (
            "checkpoint missing 'dense_motion_network'"
        )
        fresh["dense_motion_network"] = _load_relaxed(
            dense_motion_network,
            dict(ckpt["dense_motion_network"]),
            allowed_missing_prefixes=_DM_FRESH_PREFIXES,
            what="dense_motion_network",
        )

    if inpainting_network is not None:
        assert "inpainting_network" in ckpt, "checkpoint missing 'inpainting_network'"
        state = dict(ckpt["inpainting_network"])
        expanded = _expand_first_conv(
            inpainting_network.state_dict(), state, num_channels
        )
        fresh["inpainting_network"] = _load_relaxed(
            inpainting_network,
            state,
            allowed_missing_prefixes=_INP_FRESH_PREFIXES,
            what="inpainting_network",
        )
        if verbose and expanded:
            print(
                "[warmstart] inpainting first-conv expanded 3 -> "
                f"{num_channels + 4} input channels (extra channels zero-init)"
            )

    if verbose:
        for name, keys in fresh.items():
            print(
                f"[warmstart] {name}: {len(keys)} fresh (un-loaded) params "
                f"-> {sorted({k.split('.')[0] for k in keys})}"
            )
    return fresh
