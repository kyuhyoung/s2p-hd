#!/bin/bash

# S2P-HD DL Stereo Matcher Test Script
# Run inside docker container: bash usage.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/usage.log"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "$1" | tee -a "$LOGFILE"
}

> "$LOGFILE"
log ""
log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

DATA_DIR=/data/satellite/jax/jax_214_all_ba_including_config

log "${GREEN}==================================================${NC}"
log "${GREEN}  S2P-HD DL Stereo Matcher Test${NC}"
log "${GREEN}==================================================${NC}"

# ============================================================
# 1) Smoke test
# ============================================================
# Ensure /workspace's s2p-hd code is installed (overrides Docker's /home/s2p-hd copy)
# Use --rebuild flag to force reinstall (needed if C code or setup.py changed)
REBUILD=false
for arg in "$@"; do
    case $arg in
        --rebuild) REBUILD=true ;;
    esac
done

if [ "$REBUILD" = true ]; then
    log "${YELLOW}[0/3] Rebuilding s2p-hd from /workspace...${NC}"
    pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1 | tee -a "$LOGFILE"
elif python3 -c "import s2p; import os; assert '/workspace' in os.path.abspath(s2p.__file__)" 2>/dev/null; then
    log "${YELLOW}[0/3] s2p-hd already installed from /workspace (py changes auto-applied)${NC}"
else
    log "${YELLOW}[0/3] Installing s2p-hd from /workspace...${NC}"
    pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1 | tee -a "$LOGFILE"
fi

log "${YELLOW}[1/3] Smoke test...${NC}"
python3 -c "
from s2p import config, dl_stereo
cfg = config.get_default_config()
assert cfg['dl_stereo_model'] == 'monster'
print('s2p config OK')
print('dl_stereo module OK')
import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import numpy; print(f'NumPy {numpy.__version__}')
" 2>&1 | tee -a "$LOGFILE"

if [ $? -ne 0 ]; then
    log "${RED}Smoke test failed${NC}"
    exit 1
fi
log "${GREEN}Smoke test passed${NC}"

# ============================================================
# 2) Run DL stereo matcher
# ============================================================
log "${YELLOW}[2/3] Running s2p with DL stereo matcher...${NC}"
log "  Data dir: ${DATA_DIR}"
log "  Config:   config_dl_stereo.json"
log "  Output:   ${DATA_DIR}/s2p_out_dl/"

# Clean previous output to force fresh run
rm -rf ${DATA_DIR}/s2p_out_dl
log "  Cleaned previous output"

cd ${DATA_DIR}
s2p config_dl_stereo.json 2>&1 | while IFS= read -r line; do
    echo "$line"
    echo "$line" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE"
done

if [ $? -ne 0 ]; then
    log "${RED}s2p DL stereo run failed${NC}"
    exit 1
fi
log "${GREEN}DL stereo run complete${NC}"

# ============================================================
# 3) Compare with SGM baseline
# ============================================================
log "${YELLOW}[3/3] Comparing results...${NC}"

SGM_DSM="${DATA_DIR}/s2p_out/dsm.tif"
DL_DSM="${DATA_DIR}/s2p_out_dl/dsm.tif"

if [ -f "$SGM_DSM" ] && [ -f "$DL_DSM" ]; then
    python3 -c "
import numpy as np
import rasterio

with rasterio.open('${SGM_DSM}') as src:
    sgm = src.read(1).astype(np.float32)
with rasterio.open('${DL_DSM}') as src:
    dl = src.read(1).astype(np.float32)

# Align shapes
h = min(sgm.shape[0], dl.shape[0])
w = min(sgm.shape[1], dl.shape[1])
sgm, dl = sgm[:h, :w], dl[:h, :w]

# Valid pixels (both finite)
valid = np.isfinite(sgm) & np.isfinite(dl)
diff = np.abs(sgm[valid] - dl[valid])

print(f'SGM shape: {sgm.shape}, valid: {np.isfinite(sgm).sum()}')
print(f'DL  shape: {dl.shape}, valid: {np.isfinite(dl).sum()}')
print(f'Common valid pixels: {valid.sum()}')
print(f'MAE:  {np.mean(diff):.3f} m')
print(f'NMAD: {1.4826 * np.median(diff):.3f} m')
print(f'RMSE: {np.sqrt(np.mean(diff**2)):.3f} m')
" 2>&1 | tee -a "$LOGFILE"
else
    log "${YELLOW}Cannot compare: SGM=${SGM_DSM} DL=${DL_DSM}${NC}"
    [ ! -f "$SGM_DSM" ] && log "${RED}  SGM DSM not found${NC}"
    [ ! -f "$DL_DSM" ] && log "${RED}  DL DSM not found${NC}"
fi

log ""
log "${GREEN}==================================================${NC}"
log "${GREEN}  Done!${NC}"
log "${GREEN}==================================================${NC}"
