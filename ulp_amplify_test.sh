#!/bin/bash

# Inject 1-ULP noise at the interpolation step and see whether the final
# disparity diverges as much as switching kernels does.
#
#   bash ulp_amplify_test.sh
#   SIZE=1024 FRAC=0.11 bash ulp_amplify_test.sh

DOCKER_IMAGE="s2p-hd-dl:latest"
dir_data=/raid/HDD/dataset_stereo
dir_pretrained=/raid/HDD/kevin_workspace/pretrained_model

: "${PROBE_GPU:=0}"
: "${SIZE:=1536}"
: "${FRAC:=0.11}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/ulp_amplify_test.log"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"

log() { echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"; }

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== ULP perturbation amplification test (size=${SIZE}, frac=${FRAC}) ===${NC}"
log ""

docker run --rm \
    --shm-size=64g --gpus all --user root -w /workspace \
    -e CUDA_VISIBLE_DEVICES="${PROBE_GPU}" \
    -v "${SCRIPT_DIR}":/workspace \
    -v ${dir_data}:/data \
    -v ${dir_pretrained}:/pretrained \
    ${DOCKER_IMAGE} \
    bash -c "
        python3 -c 'import s2p' 2>/dev/null || pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1
        SEED=/data/satellite/jax/jax_214_all_ba_including_config/s2p_out_dl_foundation/tiles/row_0000000_height_1000/col_0000000_width_1000/pair_1
        python3 -u /workspace/tools/ulp_amplify_test.py --size ${SIZE} --frac ${FRAC} \
            --seed-ref \$SEED/rectified_ref.tif --seed-sec \$SEED/rectified_sec.tif
    " 2>&1 | stdbuf -oL tee -a "$LOGFILE"

RC=${PIPESTATUS[0]}
log ""
[ $RC -ne 0 ] && log "${RED}exited with ${RC}${NC}" || log "${GREEN}Done!${NC}"
exit $RC
