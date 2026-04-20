#!/bin/bash

# Test DL stereo with horizontal_margin=100, vertical_margin=100 on Samsung PNEO3.
# Goal: verify that increasing margin (> dl_border_trim=32) eliminates the
# "lightning" triangular NaN at tile junctions, and check seam step across
# tile columns in the last row.
#
# Run inside docker container: bash test_margin.sh
# GPU selection: defaults to GPU 7. Override: CUDA_VISIBLE_DEVICES=N bash test_margin.sh
#
# Flags:
#   --smooth-h  Enable continuous-H-field smoothing to reduce tile-boundary seams
#               (MVP: each tile blends its local H1/H2 with cached neighbor tiles').

: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

SMOOTH_H=false
SMOOTH_METHOD=correspondence
GLOBAL_H=false
for arg in "$@"; do
    case "$arg" in
        --smooth-h) SMOOTH_H=true ;;
        --log-euclidean) SMOOTH_H=true; SMOOTH_METHOD=log_euclidean ;;
        --global-h) GLOBAL_H=true ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done
if [ "$SMOOTH_H" = true ] && [ "$GLOBAL_H" = true ]; then
    echo "--smooth-h/--log-euclidean and --global-h are mutually exclusive" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_margin.log"
DATA_DIR=/data/satellite/seoul/gangnam/samsung/260406_Samseong_gwarp
if [ "$GLOBAL_H" = true ]; then
    OUT_DIR=./tiletest_1000_margin_globalH
elif [ "$SMOOTH_H" = true ]; then
    if [ "$SMOOTH_METHOD" = "log_euclidean" ]; then
        OUT_DIR=./tiletest_1000_margin_smoothH_logeuc
    else
        OUT_DIR=./tiletest_1000_margin_smoothH
    fi
else
    OUT_DIR=./tiletest_1000_margin
fi

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

# Always force-reinstall s2p from /workspace so the latest host code is
# reflected in the container's Python path. The container's base install
# points at /home/s2p-hd/ (snapshotted at Docker build), so without this
# step edits on the host do not show up at runtime. --no-deps keeps
# torch/numpy untouched. ~5-10 sec; negligible vs the 12-15 min test.
log "${YELLOW}Reinstalling s2p-hd from /workspace (editable, --force-reinstall)...${NC}"
pip3 install --root-user-action=ignore -e /workspace --force-reinstall --no-deps 2>&1 \
    | tail -3 | stdbuf -oL tee -a "$LOGFILE"

# Build C binaries if missing (not overwritten by pip install -e)
if ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Building s2p-hd C binaries...${NC}"
    make -C /workspace 2>&1 | tail -3 | stdbuf -oL tee -a "$LOGFILE"
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
  "dl_unipolarity_margin": 50,
  "dl_h_smooth": ${SMOOTH_H},
  "dl_h_smooth_method": "${SMOOTH_METHOD}",
  "dl_global_rectification": ${GLOBAL_H}
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

# Verify dl_h_smooth activation (if requested)
log ""
log "${GREEN}--- dl_h_smooth activation check ---${NC}"
if [ "$SMOOTH_H" = true ]; then
    hits=$(find "${OUT_DIR}/tiles" -name stdout.log 2>/dev/null \
           -exec grep -l '\[dl_h_smooth\]' {} + 2>/dev/null | wc -l)
    log "tiles that logged [dl_h_smooth]: ${hits} (expect 20 for full tile=1000 grid)"
    if [ "$hits" -gt 0 ]; then
        sample=$(find "${OUT_DIR}/tiles" -name stdout.log 2>/dev/null \
                 | head -3 | xargs grep -h '\[dl_h_smooth\]' 2>/dev/null | head -5)
        log "sample log lines:"
        printf "%s\n" "$sample" | while read -r line; do log "  ${line}"; done
    else
        log "${RED}WARNING: no tile logged [dl_h_smooth]. Either cfg flag was ignored"
        log "  or the s2p code that was invoked is an older build without this feature.${NC}"
        log "  Try: pip install --root-user-action=ignore -e /workspace --force-reinstall --no-deps"
    fi
else
    log "--smooth-h was NOT passed; feature disabled for this run."
fi

log ""
log "${GREEN}Done. Output in: ${DATA_DIR}/${OUT_DIR}${NC}"
