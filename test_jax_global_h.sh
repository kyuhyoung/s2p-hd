#!/bin/bash

# JAX_214: compare per-tile rectification vs global-H rectification
# on FoundationStereo across two tile_sizes, with GT-based MAE / NMAD / RMSE.
#
# Run inside docker container:
#   bash test_jax_global_h.sh
#
# GPU selection: defaults to GPU 7. Override with CUDA_VISIBLE_DEVICES=N.
#
# Outputs:
#   JAX_DATA/jax_ts{1000,600}_{pertile,globalH}/dsm.tif
#   test_jax_global_h.log               (all s2p output + final GT table)

: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_jax_global_h.log"
DATA_DIR=/data/satellite/jax/jax_214_all_ba_including_config
GT=/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_DSM_georef.tif
BASE_CONFIG=/workspace/configs/jax_214/config_dl_foundation.json

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"
log() { echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"; }

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== JAX_214 global-H vs per-tile benchmark (FoundationStereo) ===${NC}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
log "DATA_DIR=${DATA_DIR}"
log "GT=${GT}"

# Always force-reinstall so the latest host code is active.
log "${YELLOW}Reinstalling s2p-hd from /workspace (--force-reinstall)...${NC}"
pip3 install --root-user-action=ignore -e /workspace --force-reinstall --no-deps 2>&1 \
    | tail -3 | stdbuf -oL tee -a "$LOGFILE"

# Build C binaries if missing.
if ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Building s2p-hd C binaries...${NC}"
    make -C /workspace 2>&1 | tail -3 | stdbuf -oL tee -a "$LOGFILE"
fi

cd "${DATA_DIR}"

RUNS=()
for TS in 1000 600; do
    for MODE in pertile globalH; do
        OUT_REL="./jax_ts${TS}_${MODE}"
        CFG="config_jax_ts${TS}_${MODE}.json"
        USE_GLOBAL=false
        [ "$MODE" = "globalH" ] && USE_GLOBAL=true

        python3 -u -c "
import json
d = json.load(open('${BASE_CONFIG}'))
d['out_dir'] = '${OUT_REL}'
d['tile_size'] = ${TS}
d['horizontal_margin'] = 100
d['vertical_margin'] = 100
d['dl_global_rectification'] = ${USE_GLOBAL^}
json.dump(d, open('${CFG}', 'w'), indent=2)
"

        log ""
        log "${GREEN}========== ${OUT_REL}  (tile=${TS}, mode=${MODE}) ==========${NC}"
        rm -rf "${OUT_REL}"
        START=$(date +%s)
        s2p "${CFG}" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
        EXIT_CODE=${PIPESTATUS[0]}
        END=$(date +%s)
        ELAPSED=$((END - START))

        if [ $EXIT_CODE -ne 0 ] || [ ! -f "${OUT_REL}/dsm.tif" ]; then
            log "${RED}${OUT_REL}: FAILED (exit=${EXIT_CODE}, dsm.tif=$([ -f ${OUT_REL}/dsm.tif ] && echo yes || echo no))${NC}"
            RUNS+=("${OUT_REL}|FAILED|${ELAPSED}")
        else
            log "${GREEN}${OUT_REL}: ok (${ELAPSED}s)${NC}"
            RUNS+=("${OUT_REL}|ok|${ELAPSED}")
        fi
    done
done

log ""
log "${GREEN}================================================${NC}"
log "${GREEN}  GT Comparison (DL - GT, median offset removed)${NC}"
log "${GREEN}================================================${NC}"

python3 -u -c "
import os, numpy as np, rasterio
from rasterio.warp import reproject, Resampling

GT='${GT}'
runs = [
    ('tile=1000 per-tile', './jax_ts1000_pertile/dsm.tif'),
    ('tile=1000 global-H', './jax_ts1000_globalH/dsm.tif'),
    ('tile=600  per-tile', './jax_ts600_pertile/dsm.tif'),
    ('tile=600  global-H', './jax_ts600_globalH/dsm.tif'),
]

print(f'{\"Run\":<22} {\"valid\":>11} {\"MAE\":>7} {\"NMAD\":>7} {\"RMSE\":>7} {\"offset\":>7}', flush=True)
print('-'*75, flush=True)
for name, p in runs:
    if not os.path.exists(p):
        print(f'{name:<22}  (dsm.tif missing)', flush=True); continue
    with rasterio.open(p) as ps:
        pred = ps.read(1).astype(np.float32)
        meta = ps.meta.copy()
    gt_aligned = np.full(pred.shape, np.nan, dtype=np.float32)
    with rasterio.open(GT) as gs:
        reproject(source=rasterio.band(gs, 1), destination=gt_aligned,
                  dst_transform=meta['transform'], dst_crs=meta['crs'],
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    valid = np.isfinite(gt_aligned) & np.isfinite(pred)
    n = int(valid.sum())
    if n == 0:
        print(f'{name:<22}  no GT overlap', flush=True); continue
    diff = gt_aligned[valid] - pred[valid]
    off  = float(np.median(diff))
    d    = diff - off
    mae  = float(np.mean(np.abs(d)))
    nmad = float(1.4826 * np.median(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    print(f'{name:<22} {n:>11,} {mae:>7.3f} {nmad:>7.3f} {rmse:>7.3f} {off:>+7.3f}', flush=True)
" 2>&1 | stdbuf -oL tee -a "$LOGFILE"

log ""
log "${GREEN}================================================${NC}"
log "${GREEN}  Run summary${NC}"
log "${GREEN}================================================${NC}"
for r in "${RUNS[@]}"; do
    name=$(echo "$r" | cut -d'|' -f1)
    stat=$(echo "$r" | cut -d'|' -f2)
    time=$(echo "$r" | cut -d'|' -f3)
    log "${name}: ${stat} (${time}s)"
done

log ""
log "${GREEN}All done. DSMs in: ${DATA_DIR}/jax_ts*_*/dsm.tif${NC}"
