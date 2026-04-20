#!/bin/bash

# Test DL stereo with horizontal_margin=100, vertical_margin=100 on Samsung PNEO3.
# Goal: verify that increasing margin (> dl_border_trim=32) eliminates the
# "lightning" triangular NaN at tile junctions, and check seam step across
# tile columns in the last row.
#
# Run inside docker container: bash test_margin.sh
# GPU selection: defaults to GPU 7. Override: CUDA_VISIBLE_DEVICES=N bash test_margin.sh

: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_margin.log"
DATA_DIR=/data/satellite/seoul/gangnam/samsung/260406_Samseong_gwarp
OUT_DIR=./tiletest_1000_margin

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"

log() {
    echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"
}

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== Margin Test (Samsung PNEO3, FoundationStereo, tile=1000, margin=100/100) ===${NC}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# Ensure s2p-hd installed
if ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Rebuilding s2p-hd...${NC}"
    make -C /workspace clean 2>/dev/null
    pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1 | stdbuf -oL tee -a "$LOGFILE"
fi

# Ensure RGB images exist (same prep as test_tile_size.sh)
mkdir -p "${DATA_DIR}/rgb"
for src in IMG_PNEO3_STE_202603120224416_PS IMG_PNEO3_STE_202603120224586_PS; do
    dst="${DATA_DIR}/rgb/${src}.tif"
    if [ ! -f "$dst" ]; then
        log "${YELLOW}Preparing RGB: ${src}...${NC}"
        gdal_translate -b 1 -b 2 -b 3 "${DATA_DIR}/${src}.tif" "$dst" -co COMPRESS=LZW -q
        cp "${DATA_DIR}/${src}.rpc" "${DATA_DIR}/rgb/${src}.rpc"
    fi
done

cd "${DATA_DIR}"

# Write the config
CONFIG=config_test_margin.json
cat > "$CONFIG" <<EOFCFG
{
  "out_dir": "${OUT_DIR}",
  "images": [
    {"img": "rgb/IMG_PNEO3_STE_202603120224416_PS.tif"},
    {"img": "rgb/IMG_PNEO3_STE_202603120224586_PS.tif"}
  ],
  "roi": {"x": 7097, "y": 3049, "w": 4704, "h": 4080},
  "horizontal_margin": 100,
  "vertical_margin": 100,
  "tile_size": 1000,
  "disp_range_method": "sift",
  "msk_erosion": 0,
  "dsm_resolution": 0.3,
  "max_processes": 1,
  "matching_algorithm": "dl_stereo",
  "dl_stereo_model": "foundationstereo",
  "dl_stereo_ckpt": "/pretrained/foundationstereo/23-51-11/model_best_bp2.pth",
  "dl_stereo_device": "cuda:0",
  "dl_border_trim": 32,
  "dl_lr_check": false,
  "dl_lr_threshold": 2,
  "dl_unipolarity_margin": 50
}
EOFCFG

log ""
log "${GREEN}Config: ${DATA_DIR}/${CONFIG}${NC}"
log "${GREEN}Output: ${DATA_DIR}/${OUT_DIR}${NC}"
log ""

# Run
rm -rf "${OUT_DIR}"
s2p "$CONFIG" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    log "${RED}s2p failed (exit=${EXIT_CODE})${NC}"
    exit $EXIT_CODE
fi

# Quick check
if [ -f "${OUT_DIR}/dsm.tif" ]; then
    python3 -u -c "
import rasterio, numpy as np
for name in ['dsm.tif', 'dsm-filtered.tif']:
    p = '${OUT_DIR}/' + name
    try:
        with rasterio.open(p) as f:
            d = f.read(1)
            print(f'{name}: shape={d.shape} valid={np.isfinite(d).sum():,}', flush=True)
    except Exception as e:
        print(f'{name}: not generated ({e})', flush=True)
" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
else
    log "${RED}dsm.tif not generated${NC}"
fi

log ""
log "${GREEN}Done. Output in: ${DATA_DIR}/${OUT_DIR}${NC}"
