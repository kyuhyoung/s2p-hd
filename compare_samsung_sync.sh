#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/compare_samsung_sync.log"

> "$LOGFILE"

python3 -u -c "
import numpy as np
import rasterio, os
from rasterio.warp import reproject, Resampling

DATA = '/data/satellite/seoul/gangnam/samsung/260406_Samseong_gwarp'
GT = f'{DATA}/dsm.tif'

models = [
    ('SGM',                f'{DATA}/sync_sgm/dsm.tif'),
    ('Diachronic MonSter', f'{DATA}/sync_diachronic/dsm.tif'),
    ('MonSter (original)', f'{DATA}/sync_monster/dsm.tif'),
    ('FoundationStereo',   f'{DATA}/sync_foundation/dsm.tif'),
]

print('=== SYNCHRONIC (2026-03-12) vs existing DSM ===', flush=True)

results = []
for name, dsm_path in models:
    if not os.path.exists(dsm_path):
        print(f'{name}: not found', flush=True)
        results.append((name, '-', '-', '-', '-', '-'))
        continue

    with rasterio.open(dsm_path) as src:
        pred = src.read(1).astype(np.float32)
        pred_meta = src.meta.copy()
        n_pred = int(np.isfinite(pred).sum())

    gt_aligned = np.full(pred.shape, np.nan, dtype=np.float32)
    with rasterio.open(GT) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=gt_aligned,
            dst_transform=pred_meta['transform'],
            dst_crs=pred_meta['crs'],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(gt_aligned) & np.isfinite(pred)
    if valid.sum() == 0:
        results.append((name, n_pred, 0, '-', '-', '-'))
        continue

    diff = gt_aligned[valid] - pred[valid]
    offset = np.median(diff)
    diff_a = diff - offset

    mae = np.mean(np.abs(diff_a))
    nmad = 1.4826 * np.median(np.abs(diff_a))
    rmse = np.sqrt(np.mean(diff_a**2))
    results.append((name, n_pred, int(valid.sum()), f'{mae:.3f}', f'{nmad:.3f}', f'{rmse:.3f}'))

print(f'(compared against existing dsm.tif)', flush=True)
print(flush=True)
print(f'{\"Model\":<25} {\"Valid Pixels\":>12} {\"GT Overlap\":>12} {\"MAE (m)\":>10} {\"NMAD (m)\":>10} {\"RMSE (m)\":>10}', flush=True)
print('-' * 85, flush=True)
for r in results:
    print(f'{r[0]:<25} {str(r[1]):>12} {str(r[2]):>12} {str(r[3]):>10} {str(r[4]):>10} {str(r[5]):>10}', flush=True)
" 2>&1 | stdbuf -oL tee "$LOGFILE"
