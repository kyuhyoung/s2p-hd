#!/bin/bash

# Find the max tile size each of the 4 DL stereo models can handle before CUDA OOM.
#
#   bash test_max_tile.sh
#
# Runs the search inside the s2p-hd-dl docker container (host has no torch).
# Override the GPUs used with:  PROBE_GPUS=0,1 bash test_max_tile.sh

DOCKER_IMAGE="s2p-hd-dl:latest"
dir_data=/raid/HDD/dataset_stereo
dir_pretrained=/raid/HDD/kevin_workspace/pretrained_model

: "${PROBE_GPUS:=0,1,2}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/test_max_tile${LOG_TAG:-}.log"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

> "$LOGFILE"

log() {
    echo -e "$1" | stdbuf -oL tee -a "$LOGFILE"
}

log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
log "${GREEN}=== Max Tile Size Search (4 DL models) ===${NC}"
log "GPUs: ${PROBE_GPUS}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    | stdbuf -oL tee -a "$LOGFILE"

if ! docker image inspect ${DOCKER_IMAGE} &>/dev/null; then
    log "${RED}Image ${DOCKER_IMAGE} not found. Run ./using_docker.sh first.${NC}"
    exit 1
fi

log ""
log "${YELLOW}Starting search in container...${NC}"
log ""

docker run --rm \
    --shm-size=64g \
    --gpus all \
    --user root \
    -w /workspace \
    -e PROBE_GPUS="${PROBE_GPUS}" \
    -e PROBE_MODELS="${PROBE_MODELS}" \
    -e PROBE_ALLOC_CONF="${PROBE_ALLOC_CONF}" \
    -v "${SCRIPT_DIR}":/workspace \
    -v ${dir_data}:/data \
    -v ${dir_pretrained}:/pretrained \
    ${DOCKER_IMAGE} \
    bash -c '
        set -o pipefail
        if ! python3 -c "import s2p" 2>/dev/null; then
            echo "Installing s2p-hd from /workspace..."
            pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1
        elif ! python3 -c "from s2p import homography" 2>/dev/null; then
            echo "Rebuilding s2p-hd C binaries..."
            make -C /workspace clean 2>/dev/null
            pip3 install --root-user-action=ignore -e /workspace 2>&1 | tail -1
        fi
        exec python3 -u /workspace/tools/max_tile_search.py
    ' 2>&1 | stdbuf -oL tee -a "$LOGFILE"

RC=${PIPESTATUS[0]}
log ""
if [ $RC -ne 0 ]; then
    log "${RED}Search exited with code ${RC}${NC}"
else
    log "${GREEN}Done!${NC}"
fi
exit $RC
