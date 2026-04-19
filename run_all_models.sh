#!/bin/bash

# Run all DL stereo models and compare against GT
# Run inside docker container: bash run_all_models.sh
#
# GPU selection: defaults to GPU 7. Override via env var:
#   CUDA_VISIBLE_DEVICES=1 bash run_all_models.sh

: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_all_models.log"
DATA_DIR=/data/satellite/jax/jax_214_all_ba_including_config
CONFIG_DIR=${SCRIPT_DIR}/configs/jax_214

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"

log() {
    echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"
}

log "${GREEN}CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}${NC}"

# Ensure s2p-hd installed from /workspace (rebuild C binaries if needed)
if ! python3 -c "import s2p" 2>/dev/null; then
    log "${YELLOW}Installing s2p-hd from /workspace...${NC}"
    pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1 | stdbuf -oL tee -a "$LOGFILE"
elif ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Rebuilding s2p-hd C binaries...${NC}"
    make -C /workspace clean 2>/dev/null
    pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1 | stdbuf -oL tee -a "$LOGFILE"
fi

cd ${DATA_DIR}

# Configs live in repo (version-controlled). s2p is invoked from DATA_DIR so
# that relative paths in configs ('images', 'out_dir') resolve against data.
for model_cfg_name in config_dl_stereo.json config_dl_monster.json config_dl_foundation.json config_dl_stereoanywhere.json; do
    model_cfg="${CONFIG_DIR}/${model_cfg_name}"
    model_name=$(python3 -c "import json; d=json.load(open('${model_cfg}')); print(d.get('dl_stereo_model','?') + ' (' + d['out_dir'] + ')')")
    out_dir=$(python3 -c "import json; print(json.load(open('${model_cfg}'))['out_dir'])")

    log ""
    log "${GREEN}=== Running: ${model_name} ===${NC}"
    rm -rf "${out_dir}"
    s2p "${model_cfg}" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
    log "${GREEN}=== Done: ${model_name} ===${NC}"
done

# Compare all against GT
log ""
log "${GREEN}================================================${NC}"
log "${GREEN}  GT Comparison (all models)${NC}"
log "${GREEN}================================================${NC}"

python3 -u -c "
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

GT_GEOREF = '/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_DSM_georef.tif'
DATA = '/data/satellite/jax/jax_214_all_ba_including_config'

models = [
    ('SGM (s2p-hd)',            f'{DATA}/s2p_out/dsm.tif'),
    ('Diachronic MonSter',      f'{DATA}/s2p_out_dl/dsm.tif'),
    ('MonSter (original)',      f'{DATA}/s2p_out_dl_monster/dsm.tif'),
    ('FoundationStereo',        f'{DATA}/s2p_out_dl_foundation/dsm.tif'),
    ('StereoAnywhere',          f'{DATA}/s2p_out_dl_stereoanywhere/dsm.tif'),
]

results = []
for name, dsm_path in models:
    import os
    if not os.path.exists(dsm_path):
        print(f'{name}: DSM not found ({dsm_path})', flush=True)
        results.append((name, '-', '-', '-', '-', '-'))
        continue

    with rasterio.open(dsm_path) as pred_src:
        pred = pred_src.read(1).astype(np.float32)
        pred_meta = pred_src.meta.copy()

    # Reproject GT to match prediction grid
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
        print(f'{name}: no overlap with GT', flush=True)
        results.append((name, '-', '-', '-', '-', '-'))
        continue

    diff = gt_aligned[valid] - pred[valid]
    offset = np.median(diff)
    diff_aligned = diff - offset

    mae = np.mean(np.abs(diff_aligned))
    nmad = 1.4826 * np.median(np.abs(diff_aligned))
    rmse = np.sqrt(np.mean(diff_aligned**2))
    n_valid = valid.sum()
    n_pred = np.isfinite(pred).sum()

    results.append((name, n_pred, n_valid, f'{mae:.3f}', f'{nmad:.3f}', f'{rmse:.3f}'))

# Print table
print(flush=True)
print(f'{\"Model\":<25} {\"Pred Valid\":>12} {\"GT Overlap\":>12} {\"MAE (m)\":>10} {\"NMAD (m)\":>10} {\"RMSE (m)\":>10}', flush=True)
print('-' * 85, flush=True)
for r in results:
    print(f'{r[0]:<25} {str(r[1]):>12} {str(r[2]):>12} {str(r[3]):>10} {str(r[4]):>10} {str(r[5]):>10}', flush=True)
" 2>&1 | stdbuf -oL tee -a "$LOGFILE"

log ""
log "${GREEN}================================================${NC}"
log "${GREEN}  All done!${NC}"
log "${GREEN}================================================${NC}"
