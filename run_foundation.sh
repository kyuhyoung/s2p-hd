#!/bin/bash

# Run FoundationStereo and compare all models against GT
# Run inside docker container: bash run_foundation.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_foundation.log"
DATA_DIR=/data/satellite/jax/jax_214_all_ba_including_config
CONFIG_DIR=${SCRIPT_DIR}/configs/jax_214

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

> "$LOGFILE"

log() {
    echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"
}

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# Run FoundationStereo
log "${YELLOW}[1/2] Running FoundationStereo...${NC}"
cd ${DATA_DIR}
rm -rf s2p_out_dl_foundation
s2p ${CONFIG_DIR}/config_dl_foundation.json 2>&1 | stdbuf -oL tee -a "$LOGFILE"
log "${GREEN}Done${NC}"

# Compare all 4 models against GT
log ""
log "${YELLOW}[2/2] GT Comparison (all models)${NC}"

python3 -u -c "
import numpy as np
import rasterio, os
from rasterio.warp import reproject, Resampling

GT_GEOREF = '/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_DSM_georef.tif'
DATA = '/data/satellite/jax/jax_214_all_ba_including_config'

models = [
    ('SGM (s2p-hd)',            f'{DATA}/s2p_out/dsm.tif'),
    ('Diachronic MonSter',      f'{DATA}/s2p_out_dl/dsm.tif'),
    ('MonSter (original)',      f'{DATA}/s2p_out_dl_monster/dsm.tif'),
    ('FoundationStereo',        f'{DATA}/s2p_out_dl_foundation/dsm.tif'),
]

results = []
for name, dsm_path in models:
    if not os.path.exists(dsm_path):
        print(f'{name}: DSM not found', flush=True)
        results.append((name, '-', '-', '-', '-', '-'))
        continue

    with rasterio.open(dsm_path) as pred_src:
        pred = pred_src.read(1).astype(np.float32)
        pred_meta = pred_src.meta.copy()

    gt_aligned = np.full(pred.shape, np.nan, dtype=np.float32)
    with rasterio.open(GT_GEOREF) as src:
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
        results.append((name, '-', '-', '-', '-', '-'))
        continue

    diff = gt_aligned[valid] - pred[valid]
    offset = np.median(diff)
    diff_aligned = diff - offset

    mae = np.mean(np.abs(diff_aligned))
    nmad = 1.4826 * np.median(np.abs(diff_aligned))
    rmse = np.sqrt(np.mean(diff_aligned**2))
    n_pred = np.isfinite(pred).sum()
    n_valid = valid.sum()

    results.append((name, n_pred, n_valid, f'{mae:.3f}', f'{nmad:.3f}', f'{rmse:.3f}'))

print(flush=True)
print(f'{\"Model\":<25} {\"Pred Valid\":>12} {\"GT Overlap\":>12} {\"MAE (m)\":>10} {\"NMAD (m)\":>10} {\"RMSE (m)\":>10}', flush=True)
print('-' * 85, flush=True)
for r in results:
    print(f'{r[0]:<25} {str(r[1]):>12} {str(r[2]):>12} {str(r[3]):>10} {str(r[4]):>10} {str(r[5]):>10}', flush=True)
" 2>&1 | stdbuf -oL tee -a "$LOGFILE"

log ""
log "${GREEN}All done!${NC}"
