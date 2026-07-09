#!/bin/bash

# Test DL-stereo tile overlap blending on GT-backed scenes (JAX_214, JAX_260).
# Compares per-tile baseline vs per-tile + dl_overlap_blend. Expected:
# - MAE/NMAD/RMSE roughly unchanged (per-tile H preserved)
# - Tile-boundary seams reduced in the DSM
#
# Run inside docker container:
#   bash test_overlap_blend.sh

: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_overlap_blend.log"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"
log() { echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"; }

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== DL-stereo overlap blending benchmark ===${NC}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

log "${YELLOW}Reinstalling s2p-hd from /workspace...${NC}"
pip3 install --root-user-action=ignore -e /workspace --force-reinstall --no-deps 2>&1 \
    | tail -3 | stdbuf -oL tee -a "$LOGFILE"
if ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Building s2p-hd C binaries...${NC}"
    make -C /workspace 2>&1 | tail -3 | stdbuf -oL tee -a "$LOGFILE"
fi

# ---- JAX_214 ----
JAX214_DATA=/data/satellite/jax/jax_214_all_ba_including_config
JAX214_GT=/data/satellite/eonerf_dataset/truth/JAX_214/JAX_214_DSM_georef.tif
JAX214_BASE_CFG=/workspace/configs/jax_214/config_dl_foundation.json

# ---- JAX_260 ----
JAX260_DATA=/data/satellite/eonerf_dataset/image_rpc/JAX_260
JAX260_TRUTH=/data/satellite/eonerf_dataset/truth/JAX_260
JAX260_IMG_L=JAX_260_005_RGB.tif
JAX260_IMG_R=JAX_260_006_RGB.tif

# Embed RPC into JAX_260 TIFs if missing (EOnerf custom JSON -> GDAL tags)
for IMG in "${JAX260_IMG_L}" "${JAX260_IMG_R}"; do
    TIF="${JAX260_DATA}/${IMG}"
    JSON="${TIF%.tif}.json"
    HAS_RPC=$(python3 -c "import rasterio; f=rasterio.open('${TIF}'); t=f.tags(ns='RPC'); print(len(t))" 2>/dev/null || echo 0)
    if [ "$HAS_RPC" -lt 1 ] && [ -f "$JSON" ]; then
        log "${YELLOW}Injecting RPC tags into ${IMG}...${NC}"
        python3 -u <<PYEMBED 2>&1 | stdbuf -oL tee -a "$LOGFILE"
import json, rasterio
tif='${TIF}'; js='${JSON}'
with open(js) as f:
    r = json.load(f)['rpc']
tags = {
    'LINE_OFF': str(r['row_offset']), 'SAMP_OFF': str(r['col_offset']),
    'LAT_OFF': str(r['lat_offset']), 'LONG_OFF': str(r['lon_offset']),
    'HEIGHT_OFF': str(r['alt_offset']),
    'LINE_SCALE': str(r['row_scale']), 'SAMP_SCALE': str(r['col_scale']),
    'LAT_SCALE': str(r['lat_scale']), 'LONG_SCALE': str(r['lon_scale']),
    'HEIGHT_SCALE': str(r['alt_scale']),
    'LINE_NUM_COEFF': ' '.join(str(c) for c in r['row_num']),
    'LINE_DEN_COEFF': ' '.join(str(c) for c in r['row_den']),
    'SAMP_NUM_COEFF': ' '.join(str(c) for c in r['col_num']),
    'SAMP_DEN_COEFF': ' '.join(str(c) for c in r['col_den']),
}
with rasterio.open(tif, 'r+') as f:
    f.update_tags(ns='RPC', **tags)
    print(f'  {tif}: embedded {len(f.tags(ns=\"RPC\"))} RPC tags', flush=True)
PYEMBED
    fi
done

run_jax214() {
    local TS=$1 BLEND=$2 OUT_REL=$3 CFG=$4
    cd "${JAX214_DATA}"
    python3 -u -c "
import json
d = json.load(open('${JAX214_BASE_CFG}'))
d['out_dir'] = '${OUT_REL}'
d['tile_size'] = ${TS}
d['horizontal_margin'] = 100
d['vertical_margin'] = 100
d['dl_overlap_blend'] = ${BLEND^}
d['dl_global_rectification'] = False
json.dump(d, open('${CFG}', 'w'), indent=2)
"
    log ""
    log "${GREEN}========== JAX_214 ${OUT_REL} (ts=${TS}, blend=${BLEND}) ==========${NC}"
    rm -rf "${OUT_REL}"
    START=$(date +%s)
    s2p "${CFG}" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
    EXIT=${PIPESTATUS[0]}
    END=$(date +%s)
    ELAPSED=$((END - START))
    if [ $EXIT -ne 0 ] || [ ! -f "${OUT_REL}/dsm.tif" ]; then
        log "${RED}FAILED (exit=${EXIT})${NC}"
        RUNS+=("jax214|${OUT_REL}|FAILED|${ELAPSED}")
    else
        log "${GREEN}ok (${ELAPSED}s)${NC}"
        RUNS+=("jax214|${OUT_REL}|ok|${ELAPSED}")
    fi
}

run_jax260() {
    local TS=$1 BLEND=$2 OUT_REL=$3 CFG=$4
    cd "${JAX260_DATA}"
    IMG_W=$(python3 -c "import rasterio; f=rasterio.open('${JAX260_IMG_L}'); print(f.width)")
    IMG_H=$(python3 -c "import rasterio; f=rasterio.open('${JAX260_IMG_L}'); print(f.height)")
    cat > "${CFG}" <<EOFCFG
{
  "out_dir": "${OUT_REL}",
  "images": [
    {"img": "${JAX260_IMG_L}"},
    {"img": "${JAX260_IMG_R}"}
  ],
  "roi": {"x": 0, "y": 0, "w": ${IMG_W}, "h": ${IMG_H}},
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
  "dl_overlap_blend": ${BLEND},
  "dl_global_rectification": false
}
EOFCFG
    log ""
    log "${GREEN}========== JAX_260 ${OUT_REL} (ts=${TS}, blend=${BLEND}) ==========${NC}"
    rm -rf "${OUT_REL}"
    START=$(date +%s)
    s2p "${CFG}" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
    EXIT=${PIPESTATUS[0]}
    END=$(date +%s)
    ELAPSED=$((END - START))
    if [ $EXIT -ne 0 ] || [ ! -f "${OUT_REL}/dsm.tif" ]; then
        log "${RED}FAILED (exit=${EXIT})${NC}"
        RUNS+=("jax260|${OUT_REL}|FAILED|${ELAPSED}")
    else
        log "${GREEN}ok (${ELAPSED}s)${NC}"
        RUNS+=("jax260|${OUT_REL}|ok|${ELAPSED}")
    fi
}

RUNS=()
# JAX_214: tile=600 baseline vs overlap
run_jax214 600 false "./jax_ts600_overlap_baseline" "config_jax_ts600_overlap_baseline.json"
run_jax214 600 true  "./jax_ts600_overlap_blend"    "config_jax_ts600_overlap_blend.json"
# JAX_260: tile=400 baseline vs overlap
run_jax260 400 false "./jax260_ts400_overlap_baseline" "config_jax260_ts400_overlap_baseline.json"
run_jax260 400 true  "./jax260_ts400_overlap_blend"    "config_jax260_ts400_overlap_blend.json"

log ""
log "${GREEN}================================================${NC}"
log "${GREEN}  GT Comparison (DL - GT, median offset removed)${NC}"
log "${GREEN}================================================${NC}"

python3 -u -c "
import os, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
runs = [
    ('JAX_214 baseline',  '${JAX214_DATA}/jax_ts600_overlap_baseline/dsm.tif', '${JAX214_GT}'),
    ('JAX_214 blend',     '${JAX214_DATA}/jax_ts600_overlap_blend/dsm.tif',    '${JAX214_GT}'),
    ('JAX_260 baseline',  '${JAX260_DATA}/jax260_ts400_overlap_baseline/dsm.tif', '${JAX260_TRUTH}/JAX_260_DSM_georef.tif'),
    ('JAX_260 blend',     '${JAX260_DATA}/jax260_ts400_overlap_blend/dsm.tif',    '${JAX260_TRUTH}/JAX_260_DSM_georef.tif'),
]
print(f'{\"Run\":<22} {\"valid\":>11} {\"MAE\":>7} {\"NMAD\":>7} {\"RMSE\":>7} {\"offset\":>7}', flush=True)
print('-'*75, flush=True)
for name, p, gt in runs:
    if not os.path.exists(p):
        print(f'{name:<22}  (dsm.tif missing)', flush=True); continue
    with rasterio.open(p) as ps:
        pred = ps.read(1).astype(np.float32); meta = ps.meta.copy()
    gt_aligned = np.full(pred.shape, np.nan, dtype=np.float32)
    with rasterio.open(gt) as gs:
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
    scene=$(echo "$r" | cut -d'|' -f1)
    name=$(echo "$r" | cut -d'|' -f2)
    stat=$(echo "$r" | cut -d'|' -f3)
    t=$(echo "$r" | cut -d'|' -f4)
    log "${scene} ${name}: ${stat} (${t}s)"
done

log ""
log "${GREEN}All done.${NC}"
