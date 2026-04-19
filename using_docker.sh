#!/bin/bash

# S2P-HD + DL Stereo Matcher Docker Container
#
# Usage:
#   ./using_docker.sh              # build + enter
#   ./using_docker.sh --no-cache   # rebuild from scratch

DOCKER_IMAGE="s2p-hd-dl:latest"

# Directory configuration
dir_data=/raid/HDD/dataset_stereo
dir_pretrained=/raid/HDD/kevin_workspace/pretrained_model
# dir_diachronic no longer needed (thirdparty/ bundled in s2p-hd)

# Log file (same folder as script, overwritten each run)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/using_docker.log"

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

# ============================================================
# Check and download pretrained models if missing
# ============================================================
log "${YELLOW}Checking pretrained models...${NC}"
mkdir -p "${dir_pretrained}/diachronic-stereo"
mkdir -p "${dir_pretrained}/monster"
mkdir -p "${dir_pretrained}/foundationstereo/23-51-11"
mkdir -p "${dir_pretrained}/foundationstereo/11-33-40"
mkdir -p "${dir_pretrained}/stereoanywhere"
mkdir -p "${dir_pretrained}/Depth-Anything-V2-Large"

download_hf() {
    local url="$1" dst="$2" name="$3"
    if [ -f "$dst" ]; then
        log "  ${name}: already exists"
    else
        log "  ${name}: downloading..."
        wget -q --show-progress -O "$dst" "$url" 2>&1 | tee -a "$LOGFILE"
        if [ $? -ne 0 ]; then
            log "${RED}  ${name}: download failed${NC}"
            rm -f "$dst"
        else
            log "  ${name}: done ($(du -h "$dst" | cut -f1))"
        fi
    fi
}

# Diachronic MonSter (satellite fine-tuned)
download_hf \
    "https://huggingface.co/emasquil/diachronic-stereo/resolve/main/final.pth" \
    "${dir_pretrained}/diachronic-stereo/final.pth" \
    "Diachronic MonSter"

# MonSter original
download_hf \
    "https://huggingface.co/cjd24/MonSter/resolve/main/mix_all.pth" \
    "${dir_pretrained}/monster/mix_all.pth" \
    "MonSter (mix_all)"

# Depth Anything V2 Large
download_hf \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
    "${dir_pretrained}/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth" \
    "Depth Anything V2 Large"

# StereoAnywhere
download_hf \
    "https://huggingface.co/emasquil/diachronic-stereo/resolve/main/stereoanywhere_sceneflow.pth" \
    "${dir_pretrained}/stereoanywhere/stereoanywhere_sceneflow.pth" \
    "StereoAnywhere"

# FoundationStereo (large model) - Google Drive, manual download needed
if [ ! -f "${dir_pretrained}/foundationstereo/23-51-11/model_best_bp2.pth" ]; then
    log "${YELLOW}  FoundationStereo: not found. Download manually from:${NC}"
    log "    https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"
    log "    Place model_best_bp2.pth and cfg.yaml in ${dir_pretrained}/foundationstereo/23-51-11/"
else
    log "  FoundationStereo: already exists"
fi

log "${GREEN}Pretrained models check done${NC}"

# Check Dockerfile exists
if [ ! -f "${SCRIPT_DIR}/Dockerfile" ]; then
    log "${RED}Dockerfile not found. Run from s2p-hd repo root.${NC}"
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
sudo docker build $BUILD_FLAGS -f Dockerfile -t ${DOCKER_IMAGE} . 2>&1 | while IFS= read -r line; do
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
