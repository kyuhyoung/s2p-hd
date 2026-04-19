#!/bin/bash

# Test DL stereo with increasing tile sizes on Samsung PNEO3
# Run inside docker container: bash test_tile_size.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_tile_size.log"
DATA_DIR=/data/satellite/seoul/gangnam/samsung/260406_Samseong_gwarp

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"

log() {
    echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"
}

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== Tile Size Limit Test (Samsung PNEO3, FoundationStereo) ===${NC}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# Ensure s2p-hd installed
if ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Rebuilding s2p-hd...${NC}"
    make -C /workspace clean 2>/dev/null
    pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1 | stdbuf -oL tee -a "$LOGFILE"
fi

# Ensure RGB images exist
mkdir -p ${DATA_DIR}/rgb
for src in IMG_PNEO3_STE_202603120224416_PS IMG_PNEO3_STE_202603120224586_PS; do
    dst="${DATA_DIR}/rgb/${src}.tif"
    if [ ! -f "$dst" ]; then
        gdal_translate -b 1 -b 2 -b 3 "${DATA_DIR}/${src}.tif" "$dst" -co COMPRESS=LZW -q
        cp "${DATA_DIR}/${src}.rpc" "${DATA_DIR}/rgb/${src}.rpc"
    fi
done

cd ${DATA_DIR}

# Test tile sizes: 1000, 2000, 4000, 8000, full (no tiling)
for TILE_SIZE in 1000 2000 4000 8000; do
    OUT_DIR="./tiletest_${TILE_SIZE}"
    log ""
    log "${YELLOW}=== tile_size=${TILE_SIZE} ===${NC}"

    cat > config_tiletest.json << EOFCFG
{
  "out_dir": "${OUT_DIR}",
  "images": [
    {"img": "rgb/IMG_PNEO3_STE_202603120224416_PS.tif"},
    {"img": "rgb/IMG_PNEO3_STE_202603120224586_PS.tif"}
  ],
  "roi": {"x": 7097, "y": 3049, "w": 4704, "h": 4080},
  "horizontal_margin": 20,
  "vertical_margin": 5,
  "tile_size": ${TILE_SIZE},
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

    rm -rf "${OUT_DIR}"

    # Monitor VRAM during run
    nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 > /tmp/vram_before.txt

    s2p config_tiletest.json 2>&1 | stdbuf -oL tee -a "$LOGFILE"
    EXIT_CODE=$?

    nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 > /tmp/vram_after.txt

    if [ $EXIT_CODE -ne 0 ]; then
        log "${RED}FAILED at tile_size=${TILE_SIZE}${NC}"
        log "VRAM before: $(cat /tmp/vram_before.txt), after: $(cat /tmp/vram_after.txt)"
        break
    fi

    # Check result
    if [ -f "${OUT_DIR}/dsm.tif" ]; then
        python3 -u -c "
import rasterio, numpy as np
with rasterio.open('${OUT_DIR}/dsm.tif') as f:
    d = f.read(1)
    valid = np.isfinite(d).sum()
    print(f'tile_size=${TILE_SIZE}: DSM valid={valid}, shape={d.shape}', flush=True)
" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
    else
        log "${RED}tile_size=${TILE_SIZE}: DSM not generated${NC}"
        # Show errors from tile logs
        for tlog in $(find "${OUT_DIR}/tiles" -name "stdout.log" 2>/dev/null); do
            errs=$(grep -A5 "ERROR\|failed\|Traceback\|OOM\|CUDA out of memory" "$tlog" 2>/dev/null | head -20)
            if [ -n "$errs" ]; then
                log "${RED}  $(echo $tlog | sed 's|.*/tiles/||'):${NC}"
                echo "$errs" | stdbuf -oL tee -a "$LOGFILE"
            fi
        done
    fi
done

# Compare all tile size results against each other
log ""
log "${GREEN}=== Comparison ===${NC}"

python3 -u -c "
import rasterio, numpy as np, os
from rasterio.warp import reproject, Resampling

DATA = '${DATA_DIR}'
results = []

# Use tile_size=1000 as reference
ref_path = f'{DATA}/tiletest_1000/dsm.tif'
if not os.path.exists(ref_path):
    print('Reference (tile_size=1000) not found', flush=True)
    exit()

with rasterio.open(ref_path) as ref_src:
    ref = ref_src.read(1).astype(np.float32)
    ref_meta = ref_src.meta.copy()

for ts in [1000, 2000, 4000, 8000]:
    dsm_path = f'{DATA}/tiletest_{ts}/dsm.tif'
    if not os.path.exists(dsm_path):
        results.append((ts, '-', '-', '-', '-'))
        continue

    with rasterio.open(dsm_path) as src:
        pred = src.read(1).astype(np.float32)
        pred_meta = src.meta.copy()
        n_valid = int(np.isfinite(pred).sum())

    # Reproject ref to match pred
    ref_aligned = np.full(pred.shape, np.nan, dtype=np.float32)
    with rasterio.open(ref_path) as ref_src:
        reproject(
            source=rasterio.band(ref_src, 1),
            destination=ref_aligned,
            dst_transform=pred_meta['transform'],
            dst_crs=pred_meta['crs'],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(ref_aligned) & np.isfinite(pred)
    if valid.sum() == 0 or ts == 1000:
        results.append((ts, n_valid, '-', '-', '-'))
        continue

    diff = ref_aligned[valid] - pred[valid]
    offset = np.median(diff)
    diff_a = diff - offset
    mae = np.mean(np.abs(diff_a))
    nmad = 1.4826 * np.median(np.abs(diff_a))

    results.append((ts, n_valid, f'{mae:.3f}', f'{nmad:.3f}', f'{offset:.3f}'))

print(f'{\"Tile Size\":>10} {\"Valid Pixels\":>14} {\"MAE vs 1000\":>12} {\"NMAD vs 1000\":>13} {\"Offset\":>10}', flush=True)
print('-' * 65, flush=True)
for r in results:
    print(f'{str(r[0]):>10} {str(r[1]):>14} {str(r[2]):>12} {str(r[3]):>13} {str(r[4]):>10}', flush=True)
" 2>&1 | stdbuf -oL tee -a "$LOGFILE"

log ""
log "${GREEN}Done!${NC}"
