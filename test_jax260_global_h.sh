#!/bin/bash

# JAX_260: per-tile vs global-H benchmark on FoundationStereo with EO-NeRF GT.
# JAX_260 cropped images are small (~745x813) so tile_size=400 is used to
# still get >=4 tiles with boundaries.
#
# Run inside docker container:
#   bash test_jax260_global_h.sh

: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_jax260_global_h.log"
DATA_DIR=/data/satellite/eonerf_dataset/image_rpc/JAX_260
TRUTH_DIR=/data/satellite/eonerf_dataset/truth/JAX_260

IMG_L=JAX_260_005_RGB.tif
IMG_R=JAX_260_006_RGB.tif

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"
log() { echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"; }

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== JAX_260 global-H vs per-tile benchmark (FoundationStereo) ===${NC}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for f in "${DATA_DIR}/${IMG_L}" "${DATA_DIR}/${IMG_R}"; do
    if [ ! -f "$f" ]; then log "${RED}missing: ${f}${NC}"; exit 1; fi
done
GT=$(ls "${TRUTH_DIR}"/*DSM_georef.tif 2>/dev/null | head -1)
[ -z "$GT" ] && GT=$(ls "${TRUTH_DIR}"/*DSM.tif 2>/dev/null | head -1)
if [ -z "$GT" ]; then
    log "${RED}no GT DSM found in ${TRUTH_DIR}${NC}"; exit 1
fi
log "GT: ${GT}"

log "${YELLOW}Reinstalling s2p-hd from /workspace...${NC}"
pip3 install --root-user-action=ignore -e /workspace --force-reinstall --no-deps 2>&1 \
    | tail -3 | stdbuf -oL tee -a "$LOGFILE"
if ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Building s2p-hd C binaries...${NC}"
    make -C /workspace 2>&1 | tail -3 | stdbuf -oL tee -a "$LOGFILE"
fi

cd "${DATA_DIR}"

IMG_W=$(python3 -c "import rasterio; f=rasterio.open('${DATA_DIR}/${IMG_L}'); print(f.width)")
IMG_H=$(python3 -c "import rasterio; f=rasterio.open('${DATA_DIR}/${IMG_L}'); print(f.height)")
# Use full image as ROI (already small); tile_size 400 gives ~2x3 grid with boundaries.
ROI_X=0; ROI_Y=0
ROI_W=$IMG_W
ROI_H=$IMG_H
log "Image: ${IMG_W} x ${IMG_H}  ->  ROI x=${ROI_X} y=${ROI_Y} w=${ROI_W} h=${ROI_H}"

RUNS=()
for TS in 600 400; do
    for MODE in pertile globalH; do
        OUT_REL="./jax260_ts${TS}_${MODE}"
        CFG="config_jax260_ts${TS}_${MODE}.json"
        USE_GLOBAL=false
        [ "$MODE" = "globalH" ] && USE_GLOBAL=true

        cat > "${CFG}" <<EOFCFG
{
  "out_dir": "${OUT_REL}",
  "images": [
    {"img": "${IMG_L}"},
    {"img": "${IMG_R}"}
  ],
  "roi": {"x": ${ROI_X}, "y": ${ROI_Y}, "w": ${ROI_W}, "h": ${ROI_H}},
  "horizontal_margin": 100,
  "vertical_margin": 100,
  "tile_size": ${TS},
  "disp_range_method": "sift",
  "msk_erosion": 0,
  "dsm_resolution": 0.5,
  "max_processes": 1,
  "matching_algorithm": "dl_stereo",
  "dl_stereo_model": "foundationstereo",
  "dl_stereo_ckpt": "/pretrained/foundationstereo/23-51-11/model_best_bp2.pth",
  "dl_stereo_device": "cuda:0",
  "dl_border_trim": 32,
  "dl_lr_check": false,
  "dl_lr_threshold": 2,
  "dl_unipolarity_margin": 50,
  "dl_global_rectification": ${USE_GLOBAL}
}
EOFCFG
        log ""
        log "${GREEN}========== ${OUT_REL}  (tile=${TS}, mode=${MODE}) ==========${NC}"
        rm -rf "${OUT_REL}"
        START=$(date +%s)
        s2p "${CFG}" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
        EXIT=${PIPESTATUS[0]}
        END=$(date +%s)
        ELAPSED=$((END - START))

        if [ $EXIT -ne 0 ] || [ ! -f "${OUT_REL}/dsm.tif" ]; then
            log "${RED}${OUT_REL}: FAILED (exit=${EXIT})${NC}"
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
GT = '${GT}'
runs = [
    ('tile=600 per-tile', './jax260_ts600_pertile/dsm.tif'),
    ('tile=600 global-H', './jax260_ts600_globalH/dsm.tif'),
    ('tile=400 per-tile', './jax260_ts400_pertile/dsm.tif'),
    ('tile=400 global-H', './jax260_ts400_globalH/dsm.tif'),
]
print(f'{\"Run\":<22} {\"valid\":>11} {\"MAE\":>7} {\"NMAD\":>7} {\"RMSE\":>7} {\"offset\":>7}', flush=True)
print('-'*75, flush=True)
for name, p in runs:
    if not os.path.exists(p):
        print(f'{name:<22}  (dsm.tif missing)', flush=True); continue
    with rasterio.open(p) as ps:
        pred = ps.read(1).astype(np.float32); meta = ps.meta.copy()
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
    off  = float(np.median(diff)); d = diff - off
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
    t=$(echo "$r" | cut -d'|' -f3)
    log "${name}: ${stat} (${t}s)"
done

log ""
log "${GREEN}All done. DSMs in ${DATA_DIR}/jax260_ts*_*/dsm.tif${NC}"
