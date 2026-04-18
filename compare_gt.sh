#!/bin/bash

# Compare SGM and DL DSM results against GT DSM
# Run inside docker container: bash compare_gt.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/compare_gt.log"

> "$LOGFILE"

python3 -u -c "
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os

GT_DSM = '/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_DSM.tif'
GT_GEOREF = '/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_DSM_georef.tif'
GT_EPSG4326 = '/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_DSM_georef_epsg4326.tif'
GT_CLS = '/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_CLS.tif'
SGM_DSM = '/data/satellite/jax/jax_214_all_ba_including_config/s2p_out/dsm.tif'
DL_DSM = '/data/satellite/jax/jax_214_all_ba_including_config/s2p_out_dl/dsm.tif'

print('=== DSM metadata ===', flush=True)
for name, path in [('GT DSM', GT_DSM), ('GT georef', GT_GEOREF), ('GT epsg4326', GT_EPSG4326),
                    ('SGM DSM', SGM_DSM), ('DL DSM', DL_DSM)]:
    if not os.path.exists(path):
        print(f'{name}: NOT FOUND', flush=True)
        continue
    with rasterio.open(path) as src:
        print(f'{name}: crs={src.crs}, shape={src.shape}, res={src.res}', flush=True)
        print(f'  bounds={src.bounds}', flush=True)

# Determine which GT to use (match CRS with s2p output)
with rasterio.open(SGM_DSM) as src:
    pred_crs = src.crs
    pred_bounds = src.bounds
    pred_transform = src.transform
    pred_shape = src.shape

print(f'\nPredicted DSM CRS: {pred_crs}', flush=True)

# Pick the GT that matches CRS, or reproject
gt_path = None
for candidate in [GT_EPSG4326, GT_GEOREF, GT_DSM]:
    if not os.path.exists(candidate):
        continue
    with rasterio.open(candidate) as src:
        if src.crs == pred_crs:
            gt_path = candidate
            break

if gt_path is None:
    # Use epsg4326 and reproject
    gt_path = GT_EPSG4326 if os.path.exists(GT_EPSG4326) else GT_GEOREF
    print(f'No CRS match, will reproject GT from {gt_path}', flush=True)

print(f'Using GT: {gt_path}', flush=True)

def load_and_align_to_pred(dsm_path, pred_path):
    '''Load a DSM and align it to the prediction grid via reprojection.'''
    with rasterio.open(pred_path) as pred_src:
        pred_meta = pred_src.meta.copy()
        pred_data = pred_src.read(1).astype(np.float32)

    with rasterio.open(dsm_path) as src:
        if src.crs == pred_src.crs and src.transform == pred_src.transform and src.shape == pred_src.shape:
            return src.read(1).astype(np.float32), pred_data

        # Reproject DSM to match pred grid
        aligned = np.full(pred_data.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            dst_transform=pred_meta['transform'],
            dst_crs=pred_meta['crs'],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return aligned, pred_data

def compute_metrics(gt, pred, label, cls=None):
    '''Compute MAE, NMAD, RMSE between gt and pred, with optional median alignment.'''
    valid = np.isfinite(gt) & np.isfinite(pred)
    if cls is not None:
        # Exclude water (9) and foliage (5) if CLS available
        valid_filtered = valid & (cls != 9) & (cls != 5)
    else:
        valid_filtered = valid

    for mask, suffix in [(valid, ''), (valid_filtered, ' (no water/foliage)')]:
        if mask.sum() == 0:
            print(f'  {label}{suffix}: no valid pixels', flush=True)
            continue

        diff = gt[mask] - pred[mask]
        # Median alignment (remove vertical offset)
        offset = np.median(diff)
        diff_aligned = diff - offset

        mae = np.mean(np.abs(diff_aligned))
        nmad = 1.4826 * np.median(np.abs(diff_aligned))
        rmse = np.sqrt(np.mean(diff_aligned**2))
        pct_valid = 100 * mask.sum() / valid.sum() if valid.sum() > 0 else 0

        print(f'  {label}{suffix}:', flush=True)
        print(f'    valid pixels: {mask.sum()} ({pct_valid:.1f}%)', flush=True)
        print(f'    median offset: {offset:.3f} m', flush=True)
        print(f'    MAE:  {mae:.3f} m', flush=True)
        print(f'    NMAD: {nmad:.3f} m', flush=True)
        print(f'    RMSE: {rmse:.3f} m', flush=True)

# Load CLS if available
cls = None
if os.path.exists(GT_CLS):
    with rasterio.open(GT_CLS) as src:
        cls_raw = src.read(1)

print(flush=True)
print('=== SGM vs GT ===', flush=True)
gt_aligned_sgm, sgm_data = load_and_align_to_pred(gt_path, SGM_DSM)
if cls is not None:
    # Also reproject CLS to match
    cls_sgm = np.full(sgm_data.shape, 0, dtype=np.uint8)
    with rasterio.open(GT_CLS) as src:
        with rasterio.open(SGM_DSM) as pred_src:
            reproject(
                source=rasterio.band(src, 1),
                destination=cls_sgm,
                dst_transform=pred_src.transform,
                dst_crs=pred_src.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
else:
    cls_sgm = None
compute_metrics(gt_aligned_sgm, sgm_data, 'SGM', cls_sgm)

print(flush=True)
print('=== DL (Diachronic MonSter) vs GT ===', flush=True)
gt_aligned_dl, dl_data = load_and_align_to_pred(gt_path, DL_DSM)
if cls is not None:
    cls_dl = np.full(dl_data.shape, 0, dtype=np.uint8)
    with rasterio.open(GT_CLS) as src:
        with rasterio.open(DL_DSM) as pred_src:
            reproject(
                source=rasterio.band(src, 1),
                destination=cls_dl,
                dst_transform=pred_src.transform,
                dst_crs=pred_src.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
else:
    cls_dl = None
compute_metrics(gt_aligned_dl, dl_data, 'DL', cls_dl)

print(flush=True)
print('=== Summary ===', flush=True)
print('  SGM = s2p-hd original (MGM)', flush=True)
print('  DL  = s2p-hd + Diachronic MonSter fine-tune', flush=True)
" 2>&1 | stdbuf -oL tee "$LOGFILE"
