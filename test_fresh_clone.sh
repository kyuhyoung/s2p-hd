#!/bin/bash

# Test full workflow from fresh git clone
# Simulates what a new PC would do

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_fresh_clone.log"
TEST_DIR="/tmp/test_deep_s2p"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"

log() {
    echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"
}

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== Fresh Clone Test ===${NC}"

# ============================================================
# 1. Clean clone
# ============================================================
log "${YELLOW}[1/5] Fresh git clone...${NC}"
rm -rf ${TEST_DIR}
git clone git@github.com:kyuhyoung/s2p-hd.git ${TEST_DIR} 2>&1 | stdbuf -oL tee -a "$LOGFILE"
cd ${TEST_DIR}
git checkout feat/dl-stereo-matcher 2>&1 | stdbuf -oL tee -a "$LOGFILE"

# Verify key files exist
for f in Dockerfile using_docker.sh thirdparty/__init__.py s2p/dl_stereo.py s2p/config.py; do
    if [ -f "$f" ]; then
        log "  ✓ $f"
    else
        log "${RED}  ✗ $f MISSING${NC}"
    fi
done

# ============================================================
# 2. Check pretrained models (using_docker.sh downloads them)
# ============================================================
log ""
log "${YELLOW}[2/5] Check pretrained models...${NC}"
dir_pretrained=/raid/HDD/kevin_workspace/pretrained_model
for f in diachronic-stereo/final.pth monster/mix_all.pth Depth-Anything-V2-Large/depth_anything_v2_vitl.pth stereoanywhere/stereoanywhere_sceneflow.pth foundationstereo/23-51-11/model_best_bp2.pth; do
    if [ -f "${dir_pretrained}/$f" ]; then
        log "  ✓ $f"
    else
        log "${RED}  ✗ $f MISSING${NC}"
    fi
done

# ============================================================
# 3. Docker build
# ============================================================
log ""
log "${YELLOW}[3/5] Docker build...${NC}"
sudo docker build --no-cache -t s2p-hd-test:latest -f Dockerfile . 2>&1 | while IFS= read -r line; do
    echo "$line"
    echo "$line" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE"
done

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "${RED}Docker build FAILED${NC}"
    exit 1
fi
log "${GREEN}Docker build OK${NC}"

# ============================================================
# 4. Smoke test inside container
# ============================================================
log ""
log "${YELLOW}[4/5] Smoke test in container...${NC}"
sudo docker run --rm --gpus all \
    -v ${TEST_DIR}:/workspace \
    -v ${dir_pretrained}:/pretrained \
    -v /raid/HDD/dataset_stereo:/data \
    s2p-hd-test:latest \
    bash -c "
        pip3 install --root-user-action=ignore -q -e /workspace 2>&1 | tail -1
        python3 -u -c \"
from s2p import config, dl_stereo
cfg = config.get_default_config()
assert cfg['dl_stereo_model'] == 'monster'
print('s2p config OK')
print('dl_stereo module OK')
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print('thirdparty path:', dl_stereo._find_thirdparty_root())
\"
    " 2>&1 | stdbuf -oL tee -a "$LOGFILE"

# ============================================================
# 5. Quick single-tile test (FoundationStereo on JAX)
# ============================================================
log ""
log "${YELLOW}[5/5] Quick single-tile test (JAX, FoundationStereo)...${NC}"
sudo docker run --rm --gpus all --user root \
    -v ${TEST_DIR}:/workspace \
    -v ${dir_pretrained}:/pretrained \
    -v /raid/HDD/dataset_stereo:/data \
    s2p-hd-test:latest \
    bash -c '
        cd /workspace && make -j 2>&1 | tail -5
        pip3 install --root-user-action=ignore --no-deps --no-build-isolation -e /workspace 2>&1 | tail -3
        DATA=/data/satellite/jax/jax_214_all_ba_including_config
        rm -rf ${DATA}/s2p_out_clone_test
        cat > /tmp/config_clone_test.json << EOFCFG
{
  "out_dir": "${DATA}/s2p_out_clone_test",
  "images": [
    {"img": "${DATA}/JAX_214_005_RGB.tif"},
    {"img": "${DATA}/JAX_214_006_RGB.tif"}
  ],
  "roi": {"x": 0, "y": 0, "w": 1000, "h": 1000},
  "horizontal_margin": 20,
  "vertical_margin": 5,
  "tile_size": 1000,
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
  "dl_unipolarity_margin": 50
}
EOFCFG
        s2p /tmp/config_clone_test.json
        if [ -f "${DATA}/s2p_out_clone_test/dsm.tif" ]; then
            python3 -u -c "
import rasterio, numpy as np
with rasterio.open(\"${DATA}/s2p_out_clone_test/dsm.tif\") as f:
    d = f.read(1)
    valid = np.isfinite(d).sum()
    print(f\"DSM: shape={d.shape}, valid={valid}/{d.size} ({100*valid/d.size:.1f}%)\")
"
            echo "SINGLE TILE TEST: PASSED"
        else
            echo "SINGLE TILE TEST: FAILED (no dsm.tif)"
        fi
        rm -rf ${DATA}/s2p_out_clone_test
    ' 2>&1 | stdbuf -oL tee -a "$LOGFILE"

# ============================================================
# Cleanup
# ============================================================
log ""
log "${YELLOW}Cleaning up test directory...${NC}"
sudo rm -rf ${TEST_DIR}

log ""
log "${GREEN}==================================================${NC}"
log "${GREEN}  Fresh clone test complete!${NC}"
log "${GREEN}==================================================${NC}"
