#!/bin/bash

# S2P-HD + DL Stereo Matcher Docker Container
#
# Usage:
#   ./using_docker_dl.sh              # build + enter
#   ./using_docker_dl.sh --no-cache   # rebuild from scratch

DOCKER_IMAGE="s2p-hd-dl:latest"

# Directory configuration
dir_data=/raid/HDD/dataset_stereo
dir_pretrained=/raid/HDD/kevin_workspace/pretrained_model
# dir_diachronic no longer needed (thirdparty/ bundled in s2p-hd)

# Log file (same folder as script, overwritten each run)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/using_docker_dl.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Flush function: write to both console and log
log() {
    echo -e "$1" | tee -a "$LOGFILE"
}

# Clear log file
> "$LOGFILE"
log ""
log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# Parse arguments
NO_CACHE=false
for arg in "$@"; do
    case $arg in
        --no-cache) NO_CACHE=true ;;
    esac
done

log "${GREEN}==================================================${NC}"
log "${GREEN}  S2P-HD + DL Stereo Matcher Docker${NC}"
log "${GREEN}==================================================${NC}"

# Check Dockerfile exists
if [ ! -f "${SCRIPT_DIR}/Dockerfile.dl_stereo" ]; then
    log "${RED}Dockerfile.dl_stereo not found. Run from s2p-hd repo root.${NC}"
    exit 1
fi

# Build
BUILD_FLAGS=""
if [ "$NO_CACHE" = true ]; then
    BUILD_FLAGS="--no-cache"
    log "${YELLOW}Building (no cache)...${NC}"
else
    log "${YELLOW}Building...${NC}"
fi

cd "${SCRIPT_DIR}"
sudo docker build $BUILD_FLAGS -f Dockerfile.dl_stereo -t ${DOCKER_IMAGE} . 2>&1 | while IFS= read -r line; do
    echo "$line"
    echo "$line" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE"
done

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "${RED}Docker build failed${NC}"
    exit 1
fi
log "${GREEN}Build complete${NC}"

log "${YELLOW}Mounts:${NC}"
log "  Data:          ${dir_data} -> /data"
log "  Pretrained:    ${dir_pretrained} -> /pretrained"
log "  s2p-hd:        ${SCRIPT_DIR} -> /workspace"
log ""
log "${GREEN}Entering container...${NC}"
log "${GREEN}==================================================${NC}"
log ""

# Run
sudo docker run --rm -it \
    --shm-size=64g \
    --gpus all \
    --user root \
    --net=host \
    -w /workspace \
    -v ${SCRIPT_DIR}:/workspace \
    -v ${dir_data}:/data \
    -v ${dir_pretrained}:/pretrained \
    ${DOCKER_IMAGE} \
    bash
