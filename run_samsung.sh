#!/bin/bash

# Run all stereo models on Samsung PNEO3 data (synchronic + diachronic)
# Run inside docker container: bash run_samsung.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_samsung.log"
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
log "${GREEN}==================================================${NC}"
log "${GREEN}  Samsung PNEO3 Stereo Test${NC}"
log "${GREEN}==================================================${NC}"

# Ensure s2p-hd installed from /workspace
if ! python3 -c "from s2p import homography" 2>/dev/null; then
    log "${YELLOW}Rebuilding s2p-hd...${NC}"
    make -C /workspace clean 2>/dev/null
    pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1 | stdbuf -oL tee -a "$LOGFILE"
fi

cd ${DATA_DIR}

# ============================================================
# Preprocess: extract RGB (first 3 bands) from 6-band images
# ============================================================
log "${YELLOW}Preprocessing: extracting RGB from 6-band images...${NC}"
mkdir -p rgb
for src in IMG_PNEO3_STD_202402120229171_PS IMG_PNEO3_STD_202404090224535_PS IMG_PNEO3_STE_202603120224416_PS IMG_PNEO3_STE_202603120224586_PS; do
    dst="rgb/${src}.tif"
    if [ ! -f "$dst" ]; then
        log "  ${src} -> rgb/"
        gdal_translate -b 1 -b 2 -b 3 "${src}.tif" "$dst" -co COMPRESS=LZW -q
        # Copy RPC file
        cp "${src}.rpc" "rgb/${src}.rpc"
    else
        log "  ${src} already extracted, skipping"
    fi
done
log "${GREEN}RGB extraction done${NC}"

# ============================================================
# Create configs (using rgb/ images)
# ============================================================

# Synchronic: 2026-03-12 pair (same as existing dsm.tif)
cat > config_sync_sgm.json << 'EOFCFG'
{
  "out_dir": "./sync_sgm",
  "images": [
    {"img": "rgb/IMG_PNEO3_STE_202603120224416_PS.tif"},
    {"img": "rgb/IMG_PNEO3_STE_202603120224586_PS.tif"}
  ],
  "roi": {"x": 7097, "y": 3049, "w": 4704, "h": 4080},
  "horizontal_margin": 20,
  "vertical_margin": 5,
  "tile_size": 1000,
  "disp_range_method": "sift",
  "msk_erosion": 0,
  "dsm_resolution": 0.3,
  "max_processes": 1
}
EOFCFG

for model in monster diachronic foundation; do
    if [ "$model" = "monster" ]; then
        ckpt="/pretrained/monster/mix_all.pth"
        dav2="\"dl_depth_anything_v2_path\": \"/pretrained/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth\","
        dl_model="monster"
    elif [ "$model" = "diachronic" ]; then
        ckpt="/pretrained/diachronic-stereo/final.pth"
        dav2="\"dl_depth_anything_v2_path\": \"/pretrained/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth\","
        dl_model="monster"
    elif [ "$model" = "foundation" ]; then
        ckpt="/pretrained/foundationstereo/23-51-11/model_best_bp2.pth"
        dav2=""
        dl_model="foundationstereo"
    fi

    cat > "config_sync_${model}.json" << EOFCFG
{
  "out_dir": "./sync_${model}",
  "images": [
    {"img": "rgb/IMG_PNEO3_STE_202603120224416_PS.tif"},
    {"img": "rgb/IMG_PNEO3_STE_202603120224586_PS.tif"}
  ],
  "roi": {"x": 7097, "y": 3049, "w": 4704, "h": 4080},
  "horizontal_margin": 20,
  "vertical_margin": 5,
  "tile_size": 1000,
  "disp_range_method": "sift",
  "msk_erosion": 0,
  "dsm_resolution": 0.3,
  "max_processes": 1,
  "matching_algorithm": "dl_stereo",
  "dl_stereo_model": "${dl_model}",
  "dl_stereo_ckpt": "${ckpt}",
  ${dav2}
  "dl_stereo_device": "cuda:0",
  "dl_border_trim": 32,
  "dl_lr_threshold": 2,
  "dl_unipolarity_margin": 50
}
EOFCFG

    # Diachronic: 2024-02-12 vs 2026-03-12
    cat > "config_dia_${model}.json" << EOFCFG
{
  "out_dir": "./dia_${model}",
  "images": [
    {"img": "rgb/IMG_PNEO3_STD_202402120229171_PS.tif"},
    {"img": "rgb/IMG_PNEO3_STE_202603120224416_PS.tif"}
  ],
  "roi": {"x": 6469, "y": 5661, "w": 4085, "h": 3509},
  "horizontal_margin": 20,
  "vertical_margin": 5,
  "tile_size": 1000,
  "disp_range_method": "sift",
  "msk_erosion": 0,
  "dsm_resolution": 0.3,
  "max_processes": 1,
  "matching_algorithm": "dl_stereo",
  "dl_stereo_model": "${dl_model}",
  "dl_stereo_ckpt": "${ckpt}",
  ${dav2}
  "dl_stereo_device": "cuda:0",
  "dl_border_trim": 32,
  "dl_lr_threshold": 2,
  "dl_unipolarity_margin": 50
}
EOFCFG
done

# Diachronic SGM
cat > config_dia_sgm.json << 'EOFCFG'
{
  "out_dir": "./dia_sgm",
  "images": [
    {"img": "rgb/IMG_PNEO3_STD_202402120229171_PS.tif"},
    {"img": "rgb/IMG_PNEO3_STE_202603120224416_PS.tif"}
  ],
  "roi": {"x": 6469, "y": 5661, "w": 4085, "h": 3509},
  "horizontal_margin": 20,
  "vertical_margin": 5,
  "tile_size": 1000,
  "disp_range_method": "sift",
  "msk_erosion": 0,
  "dsm_resolution": 0.3,
  "max_processes": 1
}
EOFCFG

# ============================================================
# Run all configs
# ============================================================

log ""
log "${GREEN}=== SYNCHRONIC (2026-03-12 pair) ===${NC}"

for cfg in config_sync_sgm.json config_sync_diachronic.json config_sync_monster.json config_sync_foundation.json; do
    out_dir=$(python3 -c "import json; print(json.load(open('${cfg}'))['out_dir'])")
    log ""
    log "${YELLOW}Running: ${cfg} -> ${out_dir}${NC}"
    rm -rf "${out_dir}"
    s2p "${cfg}" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
done

log ""
log "${GREEN}=== DIACHRONIC (2024-02 vs 2026-03, 2yr gap) ===${NC}"

for cfg in config_dia_sgm.json config_dia_diachronic.json config_dia_monster.json config_dia_foundation.json; do
    out_dir=$(python3 -c "import json; print(json.load(open('${cfg}'))['out_dir'])")
    log ""
    log "${YELLOW}Running: ${cfg} -> ${out_dir}${NC}"
    rm -rf "${out_dir}"
    s2p "${cfg}" 2>&1 | stdbuf -oL tee -a "$LOGFILE"
done

# ============================================================
# Compare results
# ============================================================

log ""
log "${GREEN}==================================================${NC}"
log "${GREEN}  Results Comparison${NC}"
log "${GREEN}==================================================${NC}"

python3 -u -c "
import numpy as np
import rasterio, os
from rasterio.warp import reproject, Resampling

DATA = '${DATA_DIR}'
GT_DSM = f'{DATA}/dsm.tif'

def compare_dsms(models, label):
    print(f'\n=== {label} ===', flush=True)
    results = []
    for name, dsm_path in models:
        if not os.path.exists(dsm_path):
            print(f'{name}: DSM not found', flush=True)
            results.append((name, '-', '-', '-', '-'))
            continue

        with rasterio.open(dsm_path) as src:
            pred = src.read(1).astype(np.float32)
            n_pred = np.isfinite(pred).sum()
        results.append((name, n_pred, '-', '-', '-'))

    # Cross-compare between models (first valid model as reference)
    ref_name, ref_path = None, None
    for name, path in models:
        if os.path.exists(path):
            ref_name, ref_path = name, path
            break

    if ref_path is None:
        print('No valid DSMs to compare', flush=True)
        return

    with rasterio.open(ref_path) as ref_src:
        ref = ref_src.read(1).astype(np.float32)
        ref_meta = ref_src.meta.copy()

    results = []
    for name, dsm_path in models:
        if not os.path.exists(dsm_path):
            results.append((name, '-', '-', '-', '-'))
            continue

        with rasterio.open(dsm_path) as src:
            pred = src.read(1).astype(np.float32)
            pred_meta = src.meta.copy()

        n_pred = int(np.isfinite(pred).sum())

        # Reproject ref to match pred grid for comparison
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
        if valid.sum() == 0:
            results.append((name, n_pred, '-', '-', '-'))
            continue

        diff = ref_aligned[valid] - pred[valid]
        offset = np.median(diff)
        diff_a = diff - offset

        mae = np.mean(np.abs(diff_a))
        nmad = 1.4826 * np.median(np.abs(diff_a))
        rmse = np.sqrt(np.mean(diff_a**2))
        results.append((name, n_pred, f'{mae:.3f}', f'{nmad:.3f}', f'{rmse:.3f}'))

    print(f'(compared against: {ref_name})', flush=True)
    print(f'{\"Model\":<25} {\"Valid Pixels\":>12} {\"MAE (m)\":>10} {\"NMAD (m)\":>10} {\"RMSE (m)\":>10}', flush=True)
    print('-' * 72, flush=True)
    for r in results:
        print(f'{r[0]:<25} {str(r[1]):>12} {str(r[2]):>10} {str(r[3]):>10} {str(r[4]):>10}', flush=True)

sync_models = [
    ('SGM',              f'{DATA}/sync_sgm/dsm.tif'),
    ('Diachronic MonSter', f'{DATA}/sync_diachronic/dsm.tif'),
    ('MonSter (original)', f'{DATA}/sync_monster/dsm.tif'),
    ('FoundationStereo', f'{DATA}/sync_foundation/dsm.tif'),
]

dia_models = [
    ('SGM',              f'{DATA}/dia_sgm/dsm.tif'),
    ('Diachronic MonSter', f'{DATA}/dia_diachronic/dsm.tif'),
    ('MonSter (original)', f'{DATA}/dia_monster/dsm.tif'),
    ('FoundationStereo', f'{DATA}/dia_foundation/dsm.tif'),
]

compare_dsms(sync_models, 'SYNCHRONIC (2026-03-12 pair)')
compare_dsms(dia_models, 'DIACHRONIC (2024-02 vs 2026-03, 2yr gap)')
" 2>&1 | stdbuf -oL tee -a "$LOGFILE"

# ============================================================
# Collect DSM results into one folder
# ============================================================
RESULTS_DIR="${DATA_DIR}/dsm_results"
mkdir -p "${RESULTS_DIR}"

log ""
log "${YELLOW}Collecting DSM results to ${RESULTS_DIR}/${NC}"

for pair_type in sync dia; do
    for method in sgm diachronic monster foundation; do
        src="${DATA_DIR}/${pair_type}_${method}/dsm.tif"
        dst="${RESULTS_DIR}/samsung_${pair_type}_${method}.tif"
        if [ -f "$src" ]; then
            cp "$src" "$dst"
            log "  ${pair_type}_${method} -> $(basename $dst)"
        else
            log "  ${pair_type}_${method}: dsm.tif not found"
        fi
    done
done

# Also copy existing reference DSM
if [ -f "${DATA_DIR}/dsm.tif" ]; then
    cp "${DATA_DIR}/dsm.tif" "${RESULTS_DIR}/samsung_existing_gwarp.tif"
    log "  existing dsm -> samsung_existing_gwarp.tif"
fi

log ""
log "${GREEN}All done! Results in: ${RESULTS_DIR}/${NC}"
