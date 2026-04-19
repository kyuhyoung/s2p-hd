# Copyright (C) 2026
# Deep learning stereo matcher integration for s2p-hd.
# Implements the DL correlator replacement described in Deep S2P (arXiv:2603.21882).

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from s2p import common

logger = logging.getLogger(__name__)

# Path to diachronicstereo thirdparty directory.
# Adjust this if your layout differs.
_DIACHRONIC_STEREO_ROOT = None


def _find_thirdparty_root():
    """Locate the directory containing thirdparty/."""
    global _DIACHRONIC_STEREO_ROOT
    if _DIACHRONIC_STEREO_ROOT is not None:
        return _DIACHRONIC_STEREO_ROOT

    # s2p-hd/thirdparty/ (bundled) → root is s2p-hd/
    bundled = Path(__file__).resolve().parent.parent / "thirdparty"
    if (bundled / "__init__.py").exists():
        _DIACHRONIC_STEREO_ROOT = bundled.parent
        return _DIACHRONIC_STEREO_ROOT

    # External fallbacks
    for c in [
        Path(__file__).resolve().parent.parent.parent / "diachronicstereo",
        Path("/diachronicstereo"),
    ]:
        if (c / "thirdparty" / "__init__.py").exists():
            _DIACHRONIC_STEREO_ROOT = c
            return _DIACHRONIC_STEREO_ROOT

    raise FileNotFoundError(
        "Cannot find thirdparty/. Expected at s2p-hd/thirdparty/ or adjacent diachronicstereo/"
    )


def _ensure_thirdparty_on_path():
    root = _find_thirdparty_root()
    tp = str(root)
    if tp not in sys.path:
        sys.path.insert(0, tp)


# --------------- Image I/O helpers ---------------


def _read_rectified_image(path):
    """
    Read a rectified image (GeoTIFF) and return [1, 3, H, W] float32 tensor in [0, 1].
    Single-band images are replicated to 3 channels.

    Note: common.rio_read_as_array_with_nans returns (bands, H, W) or (H, W) after squeeze.
    We convert to (H, W, C) for processing, then to tensor (1, 3, H, W).
    """
    import rasterio
    with rasterio.open(path, 'r') as src:
        arr = src.read()  # (bands, H, W)
        nodata_values = src.nodatavals
    for band, nodata in zip(arr, nodata_values):
        if nodata is not None:
            band[band == nodata] = np.nan

    # arr is (bands, H, W) — transpose to (H, W, bands)
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
    # arr.ndim == 2 means single band already squeezed (shouldn't happen with src.read())

    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    # Handle channel count
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    # Normalize to [0, 1] using percentile stretch for >8-bit images
    arr = arr.astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > 255.0:
        # 16-bit or wider: use 2-98% percentile stretch
        p2, p98 = np.percentile(arr[arr > 0], [2, 98]) if (arr > 0).any() else (vmin, vmax)
        arr = np.clip((arr - p2) / (p98 - p2 + 1e-12), 0.0, 1.0)
    elif vmax > 1.0:
        arr = arr / 255.0
    # else: already in [0, 1]

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor.unsqueeze(0)  # [1, 3, H, W]


def _pad_to_multiple(x, multiple=32):
    """Symmetric replicate-pad so H & W are divisible by multiple."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)  # left, right, top, bottom
    return F.pad(x, pad, mode="replicate"), pad


def _unpad(x, pad):
    l, r, t, b = pad
    h, w = x.shape[-2], x.shape[-1]
    return x[..., t:h - b if b else h, l:w - r if r else w]


# --------------- Model loading ---------------


_loaded_model = None
_loaded_model_name = None


def load_model(cfg):
    """
    Load a DL stereo model. Caches the model so it's loaded only once.
    Returns the model object and a predict function.
    """
    global _loaded_model, _loaded_model_name

    model_name = cfg['dl_stereo_model']
    ckpt = cfg['dl_stereo_ckpt']
    device = cfg['dl_stereo_device']
    dav2_path = cfg.get('dl_depth_anything_v2_path')

    if _loaded_model is not None and _loaded_model_name == model_name:
        return _loaded_model

    if ckpt is None:
        raise ValueError("cfg['dl_stereo_ckpt'] must be set when using dl_stereo matching")

    _ensure_thirdparty_on_path()
    import thirdparty

    logger.info(f"Loading DL stereo model: {model_name} from {ckpt}")

    if model_name == 'monster':
        model = thirdparty.build_monster(
            monster_ckpt=str(ckpt),
            depth_anything_v2_path=str(dav2_path) if dav2_path else None,
            device=device,
        )
        model.eval()

    elif model_name == 'stereoanywhere':
        stereo_model, depth_model = thirdparty.build_stereoanywhere(
            stereo_ckpt=str(ckpt),
            depth_anything_v2_path=str(dav2_path) if dav2_path else None,
            device=device,
        )
        stereo_model.eval()
        depth_model.eval()
        model = (stereo_model, depth_model)

    elif model_name == 'foundationstereo':
        model = thirdparty.build_foundation_stereo(
            foundation_ckpt=str(ckpt),
            device=device,
        )
        model.eval()

    else:
        raise ValueError(f"Unknown dl_stereo_model: {model_name}")

    _loaded_model = model
    _loaded_model_name = model_name
    logger.info(f"DL stereo model loaded: {model_name}")
    return model


# --------------- Inference ---------------


@torch.no_grad()
def _predict_monster(model, imgL, imgR, device):
    """MonSter expects [0, 255] input. Returns disparity as numpy [H, W]."""
    L = (imgL * 255.0).to(device)
    R = (imgR * 255.0).to(device)
    Lp, pad = _pad_to_multiple(L, 32)
    Rp, _ = _pad_to_multiple(R, 32)
    disp = model(Lp, Rp, iters=32, test_mode=True)  # [1, 1, H, W]
    disp = _unpad(disp, pad).squeeze().cpu().numpy()
    return disp


@torch.no_grad()
def _predict_stereoanywhere(models, imgL, imgR, device):
    """StereoAnywhere expects [0, 1] input. Output sign is flipped."""
    stereo_model, depth_model = models
    L = imgL.to(device)
    R = imgR.to(device)

    # Monocular priors (no padding needed)
    B, _, H, W = L.shape
    mono_depths = depth_model.infer_image(
        torch.cat([L, R], dim=0),
        input_size_width=W,
        input_size_height=H,
    )
    md_min, md_max = mono_depths.min(), mono_depths.max()
    mono_depths = (mono_depths - md_min) / (md_max - md_min + 1e-8)
    mono_left = mono_depths[:B]
    mono_right = mono_depths[B:2 * B]

    # Pad to 32
    Lp, pad = _pad_to_multiple(L, 32)
    Rp, _ = _pad_to_multiple(R, 32)
    mlp, _ = _pad_to_multiple(mono_left, 32)
    mrp, _ = _pad_to_multiple(mono_right, 32)

    disp, _ = stereo_model(
        Lp, Rp, mlp, mrp,
        test_mode=True,
        iters=stereo_model.args.iters,
    )
    disp = -disp  # StereoAnywhere outputs negative disparities
    disp = _unpad(disp, pad).squeeze().cpu().numpy()
    return disp


@torch.no_grad()
def _predict_foundationstereo(model, imgL, imgR, device):
    """FoundationStereo expects [0, 255] input."""
    _ensure_thirdparty_on_path()
    import thirdparty

    L = (imgL * 255.0).to(device)
    R = (imgR * 255.0).to(device)

    padder = thirdparty.FsInputPadder(L.shape, divis_by=32, force_square=False)
    Lp, Rp = padder.pad(L, R)

    with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
        disp = model.run_hierachical(Lp, Rp, iters=32, test_mode=True, small_ratio=0.5)

    disp = padder.unpad(disp).squeeze().cpu().numpy()
    return disp


def _run_model(cfg, model, imgL, imgR):
    """Run a single forward pass of the DL model. Returns raw model disparity (not sign-flipped)."""
    model_name = cfg['dl_stereo_model']
    device = cfg['dl_stereo_device']

    if model_name == 'monster':
        return _predict_monster(model, imgL, imgR, device)
    elif model_name == 'stereoanywhere':
        return _predict_stereoanywhere(model, imgL, imgR, device)
    elif model_name == 'foundationstereo':
        return _predict_foundationstereo(model, imgL, imgR, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_disparity(cfg, model, rect1_path, rect2_path, return_both=False):
    """
    Run DL stereo inference on a pair of rectified images.

    Args:
        cfg: s2p config dict
        model: loaded DL model
        rect1_path: path to rectified reference image
        rect2_path: path to rectified secondary image
        return_both: if True, also return right-to-left disparity

    Returns:
        disp: numpy array [H, W] in s2p-hd convention (right_x = left_x + disp)
              Invalid pixels are NaN.
        disp_rl: (only if return_both=True) right-to-left disparity in s2p-hd convention
    """
    imgL = _read_rectified_image(rect1_path)
    imgR = _read_rectified_image(rect2_path)

    # Left-to-right
    disp_lr_raw = _run_model(cfg, model, imgL, imgR)

    # DL models output disp = x_left - x_right (positive, left-to-right convention).
    # s2p-hd expects disp such that right_x = left_x + disp.
    # Therefore: s2p_disp = -model_disp
    disp = -disp_lr_raw

    if not return_both:
        return disp

    # Right-to-left disparity via horizontal flip.
    # Simply swapping L/R doesn't work because rectification is asymmetric.
    # Instead: flip both images horizontally, run model, flip result back.
    # This reverses the disparity direction while keeping the same epipolar geometry.
    imgL_flip = torch.flip(imgL, dims=[3])  # horizontal flip
    imgR_flip = torch.flip(imgR, dims=[3])  # horizontal flip
    disp_rl_raw = _run_model(cfg, model, imgR_flip, imgL_flip)
    # Flip the disparity map back horizontally
    disp_rl_raw = np.fliplr(disp_rl_raw)
    # The flipped model outputs disparity in the opposite direction,
    # so in s2p convention: disp_rl = disp_rl_raw (same sign as model output, positive)
    # For L-R check: disp_L(x) + disp_rl(x + disp_L(x)) ≈ 0
    # disp_L is negative (s2p convention), disp_rl should be positive
    disp_rl = disp_rl_raw

    return disp, disp_rl


# --------------- Post-processing ---------------


def left_right_consistency_check(disp_left, disp_right, threshold=2):
    """
    Left-right consistency check.
    Invalidates pixels where |disp_L(x) + disp_R(x + disp_L(x))| > threshold.

    Args:
        disp_left: [H, W] disparity map (s2p convention: right_x = left_x + disp)
        disp_right: [H, W] disparity map from right-to-left
        threshold: consistency threshold in pixels

    Returns:
        disp_left with inconsistent pixels set to NaN
    """
    h, w = disp_left.shape
    out = disp_left.copy()

    # For each pixel (x, y) in left image, the corresponding pixel in right is (x + disp, y)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    mnan = np.isnan(out)
    disp_safe = np.nan_to_num(out, nan=0.0)
    X_right = np.round(np.clip(X + disp_safe, 0, w - 1)).astype(int)

    # Check consistency
    inconsistent = np.abs(disp_safe + disp_right[Y, X_right]) > threshold
    out[inconsistent] = np.nan
    out[mnan] = np.nan
    return out


def compute_disparity_map(cfg, rect1, rect2, disp_path, mask_path,
                          model, gpu_mem_manager=None):
    """
    Compute disparity map using a DL stereo matcher.
    Drop-in replacement for block_matching.compute_disparity_map().

    Args:
        cfg: s2p config dict
        rect1: path to rectified reference image
        rect2: path to rectified secondary image
        disp_path: path to output disparity map (GeoTIFF)
        mask_path: path to output rejection mask (PNG)
        model: loaded DL model
        gpu_mem_manager: GPU memory manager (for VRAM coordination)
    """
    border_trim = cfg['dl_border_trim']
    do_lr_check = cfg.get('dl_lr_check', True)
    lr_threshold = cfg.get('dl_lr_threshold', 2)

    # Run inference (both directions if L-R check enabled)
    if do_lr_check:
        disp, disp_rl = predict_disparity(cfg, model, rect1, rect2, return_both=True)
    else:
        disp = predict_disparity(cfg, model, rect1, rect2)

    # Border trim: invalidate edges (neural aperture problem)
    if border_trim > 0:
        disp[:border_trim, :] = np.nan
        disp[-border_trim:, :] = np.nan
        disp[:, :border_trim] = np.nan
        disp[:, -border_trim:] = np.nan

    # Left-right consistency check (Deep S2P Section 3.4)
    if do_lr_check:
        n_before = np.isfinite(disp).sum()
        disp = left_right_consistency_check(disp, disp_rl, threshold=lr_threshold)
        n_after = np.isfinite(disp).sum()
        logger.info(f'L-R consistency check: {n_before - n_after} pixels rejected '
                    f'({100 * (n_before - n_after) / max(n_before, 1):.1f}%), '
                    f'{n_after} remaining')

    # Create rejection mask (1 = valid, 0 = rejected)
    mask = np.isfinite(disp).astype(np.uint8)

    # Write outputs
    common.rasterio_write(disp_path, disp.astype(np.float32))
    common.rasterio_write(mask_path, mask)
